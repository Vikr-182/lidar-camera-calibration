from lavis.models import load_model_and_preprocess
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_vicuna_instruct",
    model_type="vicuna7b",
    is_eval=True,
    device="cuda:0",
)

print('Loading model done!')

