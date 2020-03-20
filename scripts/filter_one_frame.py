import pcl
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import std_msgs.msg

raw_cloud = []

def filtering_pipeline():
    # subscribe to the pointcloud
    rospy.Subscriber("/camera/depth/color/points", PointCloud2, cloud_callback)

    filtered_cloud_pub = rospy.Publisher('/cloud_filtered', PointCloud2, queue_size=1, latch=True)

    # wait for the pointcloud to come
    while not raw_cloud:
        rospy.sleep(0.5)

    filtered_cloud = pcl.PointCloud()
    filtered_cloud.from_array(np.asarray(raw_cloud, dtype=np.float32))
    rospy.loginfo("Processing pointcloud with " + str(filtered_cloud.size) + " points.")

    pass_fill = filtered_cloud.make_passthrough_filter()
    pass_fill.set_filter_field_name("z")
    pass_fill.set_filter_limits(0, 0.5)
    filtered_cloud = pass_fill.filter()
    rospy.loginfo("Pointcloud after max range filtering has: " + str(filtered_cloud.size) + " points.")

    seg = filtered_cloud.make_segmenter_normals(ksearch=50)
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_normal_distance_weight(0.1)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_max_iterations(1000)
    seg.set_distance_threshold(0.01)
    indices, model = seg.segment()
    filtered_cloud = filtered_cloud.extract(indices, negative=True)
    rospy.loginfo("PointCloud after plane filtering has: " + str(filtered_cloud.size) + " points.")

    sor = filtered_cloud.make_voxel_grid_filter()
    sor.set_leaf_size(0.005, 0.005, 0.005)
    filtered_cloud = sor.filter()
    rospy.loginfo("Downsampled Pointcloud has: " + str(filtered_cloud.size) + " points.")

    header = std_msgs.msg.Header()
    header.frame_id = "/camera_link"
    filtered_cloud_pub.publish(point_cloud2.create_cloud_xyz32(header, filtered_cloud.to_array()))


def cloud_callback(msg):
    for p in point_cloud2.read_points(msg, skip_nans=True):
        raw_cloud.append([p[0], p[1], p[2]])


if __name__ == '__main__':
    rospy.init_node('pointcloud_filtering', anonymous=True)
    try:
        filtering_pipeline()
    except rospy.ROSInterruptException:
        pass
