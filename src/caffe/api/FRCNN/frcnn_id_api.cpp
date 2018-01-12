#include "caffe/api/FRCNN/frcnn_api.hpp"
#include <iostream>

namespace FRCNN_API{

void Identifier::cosine_similarity(vector<vector<float> > &query_feature, vector<vector<float> > &gallery_features, vector<float> &sims) {
    for(int i=0; i<gallery_features.size(); i++){
        float len_query=0;
        float len_gallery=0;
        float query_dot_gallery=0;
        float sim;
        float sq;
        for(int j=0; j<gallery_features[i].size(); j++){

            len_query += query_feature[0][j]*query_feature[0][j];
            len_gallery += gallery_features[i][j]*gallery_features[i][j];
            query_dot_gallery += query_feature[0][j]*gallery_features[i][j];
        }
        sq = sqrt(len_query) * sqrt(len_gallery);
        if (sq == 0) sq=0.0000001;
        std::cout<< query_dot_gallery << " "<< sq<<std::endl;
        sim = query_dot_gallery/sq;
        sims.push_back(sim);
    }
}

void Identifier::preprocess(const cv::Mat &img_in, const int blob_idx) {
  const vector<Blob<float> *> &input_blobs = net_->input_blobs();
  CHECK(img_in.isContinuous()) << "Warning : cv::Mat img_out is not Continuous !";
  DLOG(ERROR) << "img_in (CHW) : " << img_in.channels() << ", " << img_in.rows << ", " << img_in.cols;
  input_blobs[blob_idx]->Reshape(1, img_in.channels(), img_in.rows, img_in.cols);
  float *blob_data = input_blobs[blob_idx]->mutable_cpu_data();
  const int cols = img_in.cols;
  const int rows = img_in.rows;
  for (int i = 0; i < cols * rows; i++) {
    blob_data[cols * rows * 0 + i] =
        reinterpret_cast<float*>(img_in.data)[i * 3 + 0] ;// mean_[0];
    blob_data[cols * rows * 1 + i] =
        reinterpret_cast<float*>(img_in.data)[i * 3 + 1] ;// mean_[1];
    blob_data[cols * rows * 2 + i] =
        reinterpret_cast<float*>(img_in.data)[i * 3 + 2] ;// mean_[2];
  }
}

void Identifier::Set_Model(int gpu_id) {
  caffe::Caffe::SetDevice(gpu_id);
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
  std::string proto_file             = FrcnnParam::reid_proto;
  std::string model_file             = FrcnnParam::reid_weight;

  net_.reset(new Net<float>(proto_file, caffe::TEST));
  net_->CopyTrainedLayersFrom(model_file);
  reid_mean_[0] = FrcnnParam::reid_pixel_means[0];
  reid_mean_[1] = FrcnnParam::reid_pixel_means[1];
  reid_mean_[2] = FrcnnParam::reid_pixel_means[2];

  reid_std_[0] = FrcnnParam::reid_pixel_stds[0];
  reid_std_[1] = FrcnnParam::reid_pixel_stds[1];
  reid_std_[2] = FrcnnParam::reid_pixel_stds[2];
}

vector<boost::shared_ptr<Blob<float> > > Identifier::_predict(const vector<std::string> blob_names) {
  DLOG(ERROR) << "FORWARD BEGIN";
  float loss;
  net_->Forward(&loss);
  vector<boost::shared_ptr<Blob<float> > > output;
  for (int i = 0; i < blob_names.size(); ++i) {
    output.push_back(this->net_->blob_by_name(blob_names[i]));
  }
  DLOG(ERROR) << "FORWARD END, Loss : " << loss;
  return output;
}

void Identifier::predict(vector<cv::Mat> &crop_imgs, vector<vector<float> > &norm_features) {
  cv::Mat img;
  for(int i=0; i<crop_imgs.size(); i++){
      const int height = crop_imgs[i].rows;
      const int width = crop_imgs[i].cols;
      DLOG(INFO) << "height: " << height << " width: " << width;
      crop_imgs[i].convertTo(img, CV_32FC3);
      for (int r = 0; r < img.rows; r++) {
        for (int c = 0; c < img.cols; c++) {
          int offset = (r * img.cols + c) * 3;
          reinterpret_cast<float *>(img.data)[offset + 0] -= this->reid_mean_[0]; // B
          reinterpret_cast<float *>(img.data)[offset + 1] -= this->reid_mean_[1]; // G
          reinterpret_cast<float *>(img.data)[offset + 2] -= this->reid_mean_[2]; // R

          reinterpret_cast<float *>(img.data)[offset + 0] /= this->reid_std_[0]; // B
          reinterpret_cast<float *>(img.data)[offset + 1] /= this->reid_std_[1]; // G
          reinterpret_cast<float *>(img.data)[offset + 2] /= this->reid_std_[2]; // R
        }
      }
  this->preprocess(img, 0);
  vector<std::string> blob_names(1);
  blob_names[0] = FrcnnParam::reid_feature;
  vector<boost::shared_ptr<Blob<float> > > output = this->_predict(blob_names);
  boost::shared_ptr<Blob<float> > _norm_features(output[0]);
  vector<float> feature;
  for(int i=0; i<_norm_features->count(); i++){
    feature.push_back(_norm_features->cpu_data()[i]);
  }
    norm_features.push_back(feature);
    }
  }
} // FRCNN_API
