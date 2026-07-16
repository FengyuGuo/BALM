#include "tools.hpp"
#include <ros/ros.h>
#include <Eigen/Eigenvalues>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/MarkerArray.h>
#include <random>
#include <ctime>
#include <tf/transform_broadcaster.h>
#include "bavoxel.hpp"

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <malloc.h>

#include <algorithm>

using namespace std;

int sample_step = 1;

template <typename T>
void pub_pl_func(T &pl, ros::Publisher &pub)
{
  pl.height = 1; pl.width = pl.size();
  sensor_msgs::PointCloud2 output;
  pcl::toROSMsg(pl, output);
  output.header.frame_id = "camera_init";
  output.header.stamp = ros::Time::now();
  pub.publish(output);
}

ros::Publisher pub_path, pub_test, pub_show, pub_cute, pub_pose_init, pub_position_change, pub_pose_refine;

int read_pose(vector<double> &tims, PLM(3) &rots, PLV(3) &poss, string prename)
{
  string readname = prename + "alidarPose.csv";

  cout << readname << endl;
  ifstream inFile(readname);

  if(!inFile.is_open())
  {
    printf("open fail\n"); return 0;
  }

  int pose_size = 0;
  string lineStr, str;
  Eigen::Matrix4d aff;
  vector<double> nums;

  int ord = 0;
  while(getline(inFile, lineStr))
  {
    ord++;
    stringstream ss(lineStr);
    while(getline(ss, str, ','))
      nums.push_back(stod(str));

    if(ord == 4)
    {
      for(int j=0; j<16; j++)
        aff(j) = nums[j];

      Eigen::Matrix4d affT = aff.transpose();

      rots.push_back(affT.block<3, 3>(0, 0));
      poss.push_back(affT.block<3, 1>(0, 3));
      tims.push_back(affT(3, 3)); // timestamp of the pose
      nums.clear();
      ord = 0;
      pose_size++;
    }
  }

  return pose_size;
}

void read_file(vector<IMUST> &x_buf, vector<pcl::PointCloud<PointType>::Ptr> &pl_fulls, string &prename)
{
  prename = prename + "/datas/benchmark_realworld/";

  PLV(3) poss; // position of the whole trajectory
  PLM(3) rots; // rotation of the whole trajectory
  vector<double> tims;
  int pose_size = read_pose(tims, rots, poss, prename);
  std::cout << pose_size << " pose in pose file!\n";
  for(int m=0; m<pose_size; m+=sample_step)
  {
    string filename = prename + "full" + to_string(m) + ".pcd";

    pcl::PointCloud<PointType>::Ptr pl_ptr(new pcl::PointCloud<PointType>());
    pcl::PointCloud<pcl::PointXYZI> pl_tem;
    pcl::io::loadPCDFile(filename, pl_tem);
    for(pcl::PointXYZI &pp: pl_tem.points)
    {
      PointType ap;
      ap.x = pp.x; ap.y = pp.y; ap.z = pp.z;
      ap.intensity = pp.intensity;
      pl_ptr->push_back(ap);
    }

    pl_fulls.push_back(pl_ptr);

    IMUST curr;
    curr.R = rots[m]; curr.p = poss[m]; curr.t = tims[m];
    x_buf.push_back(curr);
  }
  // size of x_buf is the same like pl_fulls
  

}

void data_show(vector<IMUST> x_buf, vector<pcl::PointCloud<PointType>::Ptr> &pl_fulls, const vector<int>& valid_index)
{
  IMUST es0 = x_buf[0];
  for(uint i=0; i<x_buf.size(); i++)
  {
    x_buf[i].p = es0.R.transpose() * (x_buf[i].p - es0.p);
    x_buf[i].R = es0.R.transpose() * x_buf[i].R;
  }
  vector<int> valid_index_verified;
  if(valid_index.size() == 0 || valid_index.size() != x_buf.size())
  {
    std::cout << "###########################################" << std::endl;
    std::cout << "Invalid valid_index size: " << valid_index.size() << std::endl;
    std::cout << "###########################################" << std::endl;
    for(int i = 0; i < x_buf.size(); i++)
    {
      valid_index_verified.push_back(1); // will omit the invalid index if size is not equal to x_buf.size()
    }
  }
  else
  {
    valid_index_verified = valid_index;
  }

  


  pcl::PointCloud<PointType> pl_send, pl_path, pcd_full;
  int pose_size = x_buf.size();
  for(int i=0; i<pose_size; i++)
  {
    if(valid_index_verified[i] == 0)
    {
      continue;
    }
    pcl::PointCloud<PointType> pl_tem = *pl_fulls[i];
    down_sampling_voxel(pl_tem, 0.05);
    pl_transform(pl_tem, x_buf[i]);
    pl_send += pl_tem;
    pcd_full += pl_tem;
    if((i%200==0 && i!=0) || i == pose_size-1)
    {
      pub_pl_func(pl_send, pub_show);
      pl_send.clear();
      sleep(0.5);
    }

    PointType ap;
    ap.x = x_buf[i].p.x();
    ap.y = x_buf[i].p.y();
    ap.z = x_buf[i].p.z();
    ap.curvature = i;
    pl_path.push_back(ap);
  }
  pcl::PCDWriter pcd_writer;
  pcd_writer.writeBinary("/home/guo/balm_ws/src/BALM/merged.pcd", pcd_full);
  pub_pl_func(pl_path, pub_path);
}

void publish_pose(vector<IMUST> x_buf, ros::Publisher &pp)
{
  geometry_msgs::PoseArray pose_array;
  pose_array.header.frame_id = "camera_init";
  pose_array.header.stamp = ros::Time::now();
  pose_array.poses.resize(x_buf.size());
  for(int i = 0; i < x_buf.size(); i++)
  {
    pose_array.poses[i].position.x = x_buf[i].p.x();
    pose_array.poses[i].position.y = x_buf[i].p.y();
    pose_array.poses[i].position.z = x_buf[i].p.z();
    Eigen::Quaterniond q(x_buf[i].R);
    pose_array.poses[i].orientation.x = q.x();
    pose_array.poses[i].orientation.y = q.y();
    pose_array.poses[i].orientation.z = q.z();
    pose_array.poses[i].orientation.w = q.w();
  }
  std::cout << "Publish pose array to " << pp.getTopic() << std::endl;
  std::cout << "Size of pose array: " << pose_array.poses.size() << std::endl;
  pp.publish(pose_array);
}

void get_pose_change(vector<IMUST> x_buf_init, vector<IMUST> x_buf, vector<Eigen::Vector3d> &position_change, vector<double> &orientation_change)
{
  position_change.resize(x_buf.size());
  orientation_change.resize(x_buf.size());
  for(int i = 0; i < x_buf.size(); i++)
  {
    position_change[i] = x_buf[i].p - x_buf_init[i].p;
    Eigen::Matrix3d delta_R = x_buf[i].R.transpose() * x_buf_init[i].R;
    Eigen::AngleAxisd aa(delta_R);
    orientation_change[i] = aa.angle();
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "benchmark2");
  ros::NodeHandle n;
  pub_test = n.advertise<sensor_msgs::PointCloud2>("/map_test", 100);
  pub_path = n.advertise<sensor_msgs::PointCloud2>("/map_path", 100);
  pub_show = n.advertise<sensor_msgs::PointCloud2>("/map_show", 100);
  pub_cute = n.advertise<sensor_msgs::PointCloud2>("/map_cute", 100);
  pub_pose_init = n.advertise<geometry_msgs::PoseArray>("/pose_init", 100);
  pub_pose_refine = n.advertise<geometry_msgs::PoseArray>("/pose_refine", 100);


  pub_position_change = n.advertise<visualization_msgs::MarkerArray>("/position_change", 100);
  string prename, ofname;
  vector<IMUST> x_buf; // pose of the whole trajectory
  vector<pcl::PointCloud<PointType>::Ptr> pl_fulls;

  n.param<double>("voxel_size", voxel_size, 1);
  n.param<int>("sample_step", sample_step, 1);
  std::cout << "sample step: " << sample_step << std::endl;
  string file_path;
  n.param<string>("file_path", file_path, "");

  read_file(x_buf, pl_fulls, file_path);

  IMUST es0 = x_buf[0]; // initial pose
  for(uint i=0; i<x_buf.size(); i++)
  {
    x_buf[i].p = es0.R.transpose() * (x_buf[i].p - es0.p); // transform the position to the initial pose
    x_buf[i].R = es0.R.transpose() * x_buf[i].R; // transform the rotation to the initial pose
  }
  // now x_buf is the pose of the whole trajectory in the initial pose frame

  vector<IMUST> x_buf_init = x_buf; // initial pose
  

  win_size = x_buf.size();
  printf("The size of poses: %d\n", win_size);
  vector<int> valid_index;
  for(int i = 0; i < x_buf.size(); i++)
  {
    valid_index.push_back(1);
  }
  data_show(x_buf, pl_fulls, valid_index);
  printf("Check the point cloud with the initial poses.\n");
  printf("If no problem, input '1' to continue or '0' to exit...\n");
  int a; cin >> a; if(a==0) exit(0);

  publish_pose(x_buf_init, pub_pose_init);

  pcl::PointCloud<PointType> pl_full, pl_surf, pl_path, pl_send;
  for(int iterCount=0; iterCount<1; iterCount++)
  { 
    // VOXEL_LOC: just a simple x, y, z in int64
    unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map; // map all voxels

    eigen_value_array[0] = 1.0 / 16; // for what?
    eigen_value_array[1] = 1.0 / 16;
    eigen_value_array[2] = 1.0 / 9;

    for(int i=0; i<win_size; i++) // win_size = 20 for now
      cut_voxel(surf_map, *pl_fulls[i], x_buf[i], i); // what is the function doing?
      // This function will iterate through all points and assign them to the corresponding voxel.
      // win_size meaning is not clear now.

    pcl::PointCloud<PointType> pl_send;
    pub_pl_func(pl_send, pub_show);

    pcl::PointCloud<PointType> pl_cent; pl_send.clear();
    VOX_HESS voxhess;
    for(auto iter=surf_map.begin(); iter!=surf_map.end() && n.ok(); iter++)
    {
      iter->second->recut(win_size);
      iter->second->tras_opt(voxhess, win_size);
      iter->second->tras_display(pl_send, win_size);
    }

    pub_pl_func(pl_send, pub_cute);
    printf("\nThe planes (point association) cut by adaptive voxelization.\n");
    printf("If the planes are too few, the optimization will be degenerated and fail.\n");
    printf("If no problem, input '1' to continue or '0' to exit...\n");
    int a; cin >> a; if(a==0) exit(0);
    pl_send.clear(); pub_pl_func(pl_send, pub_cute);

    if(voxhess.plvec_voxels.size() < 3 * x_buf.size())
    {
      printf("Initial error too large.\n");
      printf("Please loose plane determination criteria for more planes.\n");
      printf("The optimization is terminated.\n");
      exit(0);
    }

    BALM2 opt_lsv;
    opt_lsv.damping_iter(x_buf, voxhess);

    for(auto iter=surf_map.begin(); iter!=surf_map.end();)
    {
      delete iter->second;
      surf_map.erase(iter++);
    }
    surf_map.clear();

    malloc_trim(0);
  }
  publish_pose(x_buf, pub_pose_refine);

  vector<Eigen::Vector3d> position_change;
  vector<double> orientation_change;
  get_pose_change(x_buf_init, x_buf, position_change, orientation_change);

  for(int i = 0; i < position_change.size(); i++)
  {
    std::cout << "Position change: " << position_change[i].norm() << " m" << std::endl;
  }
  double sum_orientation_change_deg = 0.0;
  vector<double> orientation_change_deg;
  vector<int> final_valid_index;
  for(int i = 0; i < orientation_change.size(); i++)
  {
    std::cout << "Orientation change: " << orientation_change[i] / 3.1415926 * 180 << " deg" << std::endl;
    orientation_change_deg.push_back(orientation_change[i] / 3.1415926 * 180);
    sum_orientation_change_deg += orientation_change_deg[i];
  }

  double avg_orientation_change_deg = sum_orientation_change_deg / orientation_change_deg.size();
  std::cout << "Average orientation change: " << avg_orientation_change_deg << " deg" << std::endl;

  double std_orientation_change_deg = 0.0;
  for(int i = 0; i < orientation_change_deg.size(); i++)
  {
    std_orientation_change_deg += (orientation_change_deg[i] - avg_orientation_change_deg) * (orientation_change_deg[i] - avg_orientation_change_deg);

    if(i < 8)
    {
      final_valid_index.push_back(0);
    }
    else
    {
      final_valid_index.push_back(1);
    }
  }
  std_orientation_change_deg = sqrt(std_orientation_change_deg / orientation_change_deg.size());
  std::cout << "Standard deviation of orientation change: " << std_orientation_change_deg << " deg" << std::endl;

  std::sort(orientation_change_deg.begin(), orientation_change_deg.end());
  double median_orientation_change_deg = orientation_change_deg[orientation_change_deg.size() / 2];
  std::cout << "Median orientation change: " << median_orientation_change_deg << " deg" << std::endl;


  printf("\nRefined point cloud is publishing...\n");
  malloc_trim(0);
  data_show(x_buf, pl_fulls, final_valid_index);
  printf("\nRefined point cloud is published.\n");

  ros::spin();
  return 0;

}


