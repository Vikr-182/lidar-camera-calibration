import laspy
import numpy as np

# 0. Creating some dummy data
my_data_xx, my_data_yy = np.meshgrid(np.linspace(-20, 20, 15), np.linspace(-20, 20, 15))
my_data_zz = my_data_xx ** 2 + 0.25 * my_data_yy ** 2
my_data = np.hstack((my_data_xx.reshape((-1, 1)), my_data_yy.reshape((-1, 1)), my_data_zz.reshape((-1, 1))))

# 1. Create a new header
header = laspy.LasHeader(point_format=3, version="1.2")
header.offsets = np.min(my_data, axis=0)
header.scales = np.array([1, 1, 1])

# 3. Create a LasWriter and a point record, then write it
with laspy.open("somepath.las", mode="w", header=header) as writer:
    point_record = laspy.ScaleAwarePointRecord.zeros(my_data.shape[0], header=header)
    point_record.x = my_data[:, 0]
    point_record.y = my_data[:, 1]
    point_record.z = my_data[:, 2]

    writer.write_points(point_record)

