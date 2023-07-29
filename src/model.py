import torch
import torch.nn as nn


from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.models.distributions import DistributionEncoder, DistributionModule
from src.models.planning_model import Planning
from src.models.temporal_model import TemporalModelIdentity, TemporalModel
from src.models.future_prediction import FuturePrediction

from src.utils.network import pack_sequence_dim, unpack_sequence_dim, set_bn_momentum
from src.utils.geometry import calculate_birds_eye_view_parameters, VoxelsSumming, pose_vec2mat

class STP3(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder_downsample = self.cfg.MODEL.ENCODER.DOWNSAMPLE
        self.encoder_out_channels = self.cfg.MODEL.ENCODER.OUT_CHANNELS
        self.latent_dim = self.cfg.MODEL.DISTRIBUTION.LATENT_DIM

        self.bev_resolution, self.bev_start_position, self.bev_dimension = calculate_birds_eye_view_parameters(
            self.cfg.LIFT.X_BOUND, self.cfg.LIFT.Y_BOUND, self.cfg.LIFT.Z_BOUND
        )
        self.bev_size = (self.bev_dimension[0].item(), self.bev_dimension[1].item()) # (200, 200)
        self.n_future = self.cfg.N_FUTURE_FRAMES # 4 frames into the future

        self.frustum = self.create_frustum()
        self.D, self.H, self.W, _ = self.frustum.shape 
        self.receptive_field = self.cfg.TIME_RECEPTIVE_FIELD # temporal context input -> 3 frames

        # lift-splat encoder
        self.encoder = Encoder(self.cfg.MODEL.ENCODER, self.D)

        # temporal model
        self.temporal_model = TemporalModel(
            in_channels=self.encoder_out_channels, 
            receptive_field=self.receptive_field,
            input_shape=self.bev_size,
            start_out_channels=self.cfg.MODEL.TEMPORAL_MODEL.START_OUT_CHANNELS,
            extra_in_channels=self.cfg.MODEL.TEMPORAL_MODEL.EXTRA_IN_CHANNELS,
            n_spatial_layers_between_temporal_layers=self.cfg.MODEL.TEMPORAL_MODEL.INBETWEEN_LAYERS,
            use_pyramid_pooling=self.cfg.MODEL.TEMPORAL_MODEL.PYRAMID_POOLING,
            )

        # future pred channels
        self.present_distribution = DistributionModule(
            in_channels=self.temporal_model.out_channels, 
            latent_dim=self.latent_dim, 
            method=self.cfg.PROBABILISTIC.METHOD)

        # future prediction
        self.future_prediction = FuturePrediction(
            in_channels=self.temporal_model.out_channels,
            latent_dim=self.latent_dim, 
            n_future=self.n_future,
            n_gru_blocks=self.cfg.MODEL.FUTURE_PRED.N_GRU_BLOCKS, 
            n_res_layers=self.cfg.MODEL.FUTURE_PRED.N_RES_LAYERS)

        # decoder
        self.decoder = Decoder(
            in_channels=self.temporal_model.out_channels,
            n_classes=len(self.cfg.SEMANTIC_SEG.VEHICLE.WEIGHTS),
            n_present=self.receptive_field,
            n_hdmap=len(self.cfg.SEMANTIC_SEG.HDMAP.ELEMENTS),
            predict_gate= {
                'perceive_hdmap': self.cfg.SEMANTIC_SEG.HDMAP.ENABLED,
                'predict_pedestrian': self.cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED,
                'predict_instance': self.cfg.INSTANCE_SEG.ENABLED,
                'predict_future_flow': self.cfg.INSTANCE_FLOW.ENABLED,
                'planning': self.cfg.PLANNING.ENABLED
            }
        )

        # planning stack
        self.planning = Planning(
            cfg=cfg, 
            feature_channel=self.encoder_out_channels, 
            gru_input_size=(self.receptive_field + self.n_future - 1),
            gru_state_size= self.cfg.PLANNING.GRU_STATE_SIZE)

        set_bn_momentum(self, self.cfg.MODEL.BN_MOMENTUM)            

    def create_frustum(self):
        # Create grid in image plane
        h, w = self.cfg.IMAGE.FINAL_DIM
        downsampled_h, downsampled_w = h // self.encoder_downsample, w // self.encoder_downsample

        # Depth grid
        depth_grid = torch.arange(*self.cfg.LIFT.D_BOUND, dtype=torch.float)
        depth_grid = depth_grid.view(-1, 1, 1).expand(-1, downsampled_h, downsampled_w)
        n_depth_slices = depth_grid.shape[0]

        # x and y grids
        x_grid = torch.linspace(0, w - 1, downsampled_w, dtype=torch.float)
        x_grid = x_grid.view(1, 1, downsampled_w).expand(n_depth_slices, downsampled_h, downsampled_w)
        y_grid = torch.linspace(0, h - 1, downsampled_h, dtype=torch.float)
        y_grid = y_grid.view(1, downsampled_h, 1).expand(n_depth_slices, downsampled_h, downsampled_w)

        # Dimension (n_depth_slices, downsampled_h, downsampled_w, 3)
        # containing data points in the image: left-right, top-bottom, depth
        frustum = torch.stack((x_grid, y_grid, depth_grid), -1)
        return nn.Parameter(frustum, requires_grad=False)

