#include <vector>
#include <string>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/FRCNN/util/frcnn_param.hpp"
#include "caffe/FRCNN/util/frcnn_helper.hpp"
#include "caffe/util/math_functions.hpp"

namespace FRCNN_API{
using std::vector;
using caffe::Blob;
using caffe::Net;
using caffe::Frcnn::FrcnnParam;
using caffe::Frcnn::Point4f;
using caffe::Frcnn::BBox;
using caffe::caffe_copy;

class Detector {
public:

  void Set_Model(int gpu_id);
  void predict(const cv::Mat &img_in, vector<BBox<float> > &results, boost::shared_ptr<Blob<float> > &img_crop);
  void predict_original(const cv::Mat &img_in, vector<BBox<float> > &results, boost::shared_ptr<Blob<float> > &img_crop);
  void predict_iterative(const cv::Mat &img_in, vector<BBox<float> > &results);

//  static Detector* getInstance();
//  static pthread_mutex_t mutex;
//protected:
  Detector(int gpu_id){
//   pthread_mutex_init(&mutex,NULL);
   Set_Model(gpu_id);
  }

private:
  void preprocess(const cv::Mat &img_in, const int blob_idx);
  void preprocess(const vector<float> &data, const int blob_idx);
  vector<boost::shared_ptr<Blob<float> > > predict(const vector<std::string> blob_names);
  boost::shared_ptr<Net<float> > net_;
  float mean_[3];
  float std_[3];
  float reid_mean_[3];
  float reid_std_[3];
  int roi_pool_layer;

//  Detector(){}
//  Detector(const Detector&);
//  Detector& operator= (const Detector&);
//  static Detector* detector;
};

class Identifier {
public:
  void Set_Model(int gpu_id);
  void predict(vector<cv::Mat> &crop_imgs, vector<vector<float> > &norm_features);
  void cosine_similarity(vector<vector<float> > &query_feature, vector<vector<float> > &gallery_features, vector<float> &sims);
  Identifier(int gpu_id){
   Set_Model(gpu_id);
  }
private:
  void preprocess(const cv::Mat &img_in, const int blob_idx);
  vector<boost::shared_ptr<Blob<float> > > _predict(const vector<std::string> blob_names);
  boost::shared_ptr<Net<float> > net_;
  float reid_mean_[3];
  float reid_std_[3];
//  Identifier(const Identifier&);
//  Identifier& operator= (const Identifier&);
//  static Identifier* identifier;

};

}
