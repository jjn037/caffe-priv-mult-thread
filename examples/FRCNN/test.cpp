#include <gflags/gflags.h>
#include <glog/logging.h>
#include "boost/algorithm/string.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/signal_handler.h"
#include "caffe/FRCNN/util/frcnn_vis.hpp"
#include "caffe/api/api.hpp"
#include <iostream>
#include <time.h>
#include <thread>
#include <mutex>
#include<queue>

#include <unistd.h>

#include <cstdlib>
#include <condition_variable>
#include "caffe/KCF/kcftracker.hpp"

using caffe::Blob;
using namespace std;
using namespace cv;

mutex tracker_mtx2;
condition_variable tracker_sleep;

mutex arg_mtx;
condition_variable _arg;

mutex mtx;
mutex in1_mtx;
mutex in2_mtx;
//std::mutex mtx2;
condition_variable q1_empty;
condition_variable q2_empty;
condition_variable q1_in_empty;
condition_variable q2_in_empty;

queue<Mat> all_in_A;
queue<Mat> all_in_B;
queue<Mat> q1_in;
queue<Mat> q2_in;
queue<vector<vector<float>>> q1_out;
queue<vector<vector<float>>> q2_out;
vector<Mat> query_img;
vector<vector<float> > query_feature;
//tracker
Rect obj_A(-1,-1,-1,-1);
Rect obj_B(-1,-1,-1,-1);
condition_variable trackerA_frames_empty;
condition_variable trackerB_frames_empty;
mutex trackerA_mtx;
mutex trackerB_mtx;
bool tracker_flag;

mutex tracker_mtx;


//id
condition_variable query_empty;
mutex id_mtx;
bool get_query_img = true;

//debug
int jj=0;

//arg_control
condition_variable img_assign_thread;
bool request_video=false;

//tracker B
queue<Mat> all_B;

string _root = "/home/d09/workspace/jjn/";

Size input_geometry_ = cv::Size(224, 224);


inline std::string INT(float x) { char A[100]; sprintf(A,"%.1f",x); return std::string(A);};
inline std::string FloatToString(float x) { char A[100]; sprintf(A,"%.4f",x); return std::string(A);};


void detector_task1(int gpu_id, queue<Mat> &q_in, queue<vector<vector<float>>> &q_out){
    API::Detector detector(gpu_id);
    int i=0;
    while(1) {
        unique_lock<mutex> lock(in1_mtx);
        i++;
        cout << "thread1"  << " "<<i<<endl;
        if(!q_in.empty()){
            cout<< "thread1"<<"ok"<<i<<endl;
          vector<vector<float> > targets;
          Mat cv_image = q_in.front();
          q_in.pop();
          stringstream newstr;
          newstr<<i;
//          imwrite( "/home/d09/workspace/jjn/" + newstr.str() + ".jpg", cv_image );
          vector<caffe::Frcnn::BBox<float> > results;
          boost::shared_ptr<Blob<float> > img_crop;
          detector.predict(cv_image, results, img_crop);
          Mat crop_img;
          vector<Mat> crop_imgs;
          cout << results.size() << endl;
          for (size_t obj = 0; obj < results.size(); obj++) {
            crop_img = cv_image(Range(int(results[obj][1]), int(results[obj][3])), Range(int(results[obj][0]), int(results[obj][2])));
            vector<float>target;
            for(int i=0; i<4; i++){target.push_back(results[obj][i]);}
            targets.push_back(target);
            resize(crop_img, crop_img, input_geometry_);
            crop_imgs.push_back(crop_img);
          }
          q_out.push(targets);
          if(!(q_out.empty())){
            q1_empty.notify_all();
          }
          cout << q1_out.size() << "out1"<< " "<< i <<endl;
          cout << q1_in.size() << "in1"<< " "<< i <<endl;
        }
        else q1_in_empty.wait(lock);
    }
}

void detector_task2(int gpu_id, queue<Mat> &q_in, queue<vector<vector<float>>> &q_out){
    API::Detector detector(gpu_id);
    int i=0;
    while(1) {
        unique_lock<mutex> lock(in2_mtx);
        i++;
        cout << "thread2" << endl;
        if(!q_in.empty()){
          vector<vector<float> > targets;
          Mat cv_image = q_in.front();
          q_in.pop();
          stringstream newstr;
          newstr<<i;
//          imwrite( "/home/d09/workspace/jjn/" + newstr.str() + ".jpg", cv_image );
          vector<caffe::Frcnn::BBox<float> > results;
          boost::shared_ptr<Blob<float> > img_crop;
          detector.predict(cv_image, results, img_crop);
          Mat crop_img;
          vector<Mat> crop_imgs;
          cout << results.size() << endl;
          for (size_t obj = 0; obj < results.size(); obj++) {
//            std::cout << "ok" << results[obj].id << "  " << INT(results[obj][0]) << " " << INT(results[obj][1]) << " " << INT(results[obj][2]) << " " << INT(results[obj][3]) << "     " << FloatToString(results[obj].confidence) << std::endl;
            crop_img = cv_image(cv::Range(int(results[obj][1]), int(results[obj][3])), cv::Range(int(results[obj][0]), int(results[obj][2])));
            vector<float>target;
            for(int i=0; i<4; i++){target.push_back(results[obj][i]);}
                targets.push_back(target);
            resize(crop_img, crop_img, input_geometry_);
            crop_imgs.push_back(crop_img);
                }
            q_out.push(targets);
            if(!(q_out.empty())){
                q2_empty.notify_all();
            }
            cout << q2_out.size() << "out2"<<endl;
            cout << q2_in.size() << "in2"<<endl;
         }
         else q2_in_empty.wait(lock);
    }
}

void identifier_task0(int gpu_id){
    // 如果接收到query_img 设置一个标志位(get_query_img) 让框画在画面上  此处保留一默认query_img
    Mat query_image = imread("/home/d09/workspace/jjn/r.png");

    query_img.push_back(query_image);
    API::Identifier identifier(gpu_id);
    if(get_query_img){
        identifier.predict(query_img, query_feature);
        query_empty.notify_all();
//        cout << "id_task0" << endl;
//        if(get_query_img){
//        break;
//        }
        get_query_img=false;
    }
}

void identifier_task1(int gpu_id){
    API::Identifier identifier(gpu_id);
    int i=0;
    vector<vector<float>> targets;
    vector<float> target;
    Mat cv_imageA;
    Mat cv_imageB;
    bool tracker_init = true;

    while(1){
        unique_lock <mutex> tracker_lock(tracker_mtx2);
        cout << "id_task1" << endl;
        if(0==(i%2)){                             //q1_out q2_out 为空
            unique_lock <mutex> lck(mtx);
            if(q1_out.empty()) q1_empty.wait(lck);
            targets = q1_out.front();
            q1_out.pop();
        }
        else{
            unique_lock <mutex> lck(mtx);
            if(q2_out.empty()) q2_empty.wait(lck);
            targets = q2_out.front();
            q2_out.pop();
        }
        i++;
        cv_imageA = all_in_A.front();
        cv_imageB = all_in_B.front();
        vector<Mat> gallery_imgs;
        Mat gallery_img;
        for(int j=0; j<targets.size();j++){
            target = targets[j];
            gallery_img = cv_imageA(Range(int(target[1]), int(target[3])), Range(int(target[0]), int(target[2])));
//            cout << int(target[1]) << " " << int(target[3]) << " " <<int(target[0]) << " " <<int(target[2]) << " " <<endl;
            gallery_imgs.push_back(gallery_img);
        }
        vector<vector<float> > gallery_features;
        identifier.predict(gallery_imgs, gallery_features);
        vector<float> sims;
        float sim_thresh=0.7;
        cout << query_feature[0].size() << endl;
        unique_lock <mutex> query_lck(id_mtx);
        cout << "query_feature size" << " " << query_feature.size() << endl;
        if(!query_feature.size()) query_empty.wait(query_lck);
        identifier.cosine_similarity(query_feature, gallery_features, sims);
//        std::cout<< sims.size()<<" " <<gallery_features.size()<< std::endl;
        cout << "consumer" << endl;
        int obj_index=-1;

        stringstream str_j;
        str_j<<jj;
        jj++;
        Scalar color = CV_RGB(0,255,255);
        for(int i=0; i<sims.size(); i++){
            Point pt(targets[i][0],targets[i][1]-5);
            ostringstream newstr;
            newstr<<sims[i];

            rectangle(cv_imageA, Point(targets[i][0], targets[i][1]), Point(targets[i][2], targets[i][3]),
                      cvScalar(0, 255, 0), 2, 8);
            putText(cv_imageA, newstr.str(), pt, CV_FONT_HERSHEY_DUPLEX, 1.0f, color,2);
            cout<< sims[i]<< endl;

            vector<float>::iterator biggest = max_element(begin(sims), end(sims));
            if(*biggest > sim_thresh){
                obj_index = distance(begin(sims), biggest);
                query_feature.clear();
                query_feature.push_back(gallery_features[obj_index]);
            }
        }
//        imshow( "ImageA", cv_imageA );
//        waitKey(1);
//        imshow( "ImageB", cv_imageB );
//        waitKey(1);
        if(obj_index>0 && tracker_init){
            int x = int(targets[obj_index][0]);
            int y = int(targets[obj_index][1]);
            int height = int(targets[obj_index][3] - targets[obj_index][1]);
            int width = int(targets[obj_index][2] - targets[obj_index][0]);
            obj_A = cvRect(x, y, width, height);
            obj_B = obj_A;
            cout << obj_A.area()<<"OBJjjjjjjjjjjjjjj"<<endl;
            trackerA_frames_empty.notify_all();                           //query feature 更新  得到query_img后 设置get_query_img开始更新query_feature
            tracker_sleep.wait(tracker_lock) ;                                                       // 停止该线程 是否需要加个线程锁
        }
        all_in_A.pop();
        all_in_B.pop();
   }
}

// tracker 初始化条件 锁条件 帧存储
void tracker_taskA(Rect &obj, queue<Mat> &frames){
    bool HOG = true;
    bool FIXEDWINDOW = false;
    bool MULTISCALE = true;
    bool LAB = true;
    bool init_tracker = true;
    KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);
    Rect result;
    Mat frame;
    int i=0;

    while(1){
        unique_lock<mutex> lock(tracker_mtx);
        if(obj.area()!=1 && !(frames.empty())){
        cout<<all_in_A.size()<<endl;
            frame = all_in_A.front();
            all_in_A.pop();
            if(init_tracker){
                tracker.init(obj, frame);
                rectangle(frame, Point(obj.x, obj.y), Point(obj.x + obj.width, obj.y + obj.height),
                      cvScalar(0, 255, 0), 2, 8);
                init_tracker=false;
            }
            else {
                result = tracker.update(frame);
                rectangle(frame, Point(result.x, result.y), Point(result.x + result.width, result.y + result.height), Scalar(0, 255, 0), 2, 8);
            }
            stringstream newstr;
            newstr<<i;
            i++;
             if(i==30) {
               cout<<"o           o  ooooooooooooooooooooooooooooooooooooooooooooooo cd cajpg"<<endl;
               _arg.notify_all();}
//            imshow("ImageA", frame);
            waitKey(1);
            cout<<"/home/d09/workspace/jjn/tt" + newstr.str() + ".jpg"<<endl;
//            imwrite( "/home/d09/workspace/jjn/tt" + newstr.str() + ".jpg", frame);
        }
        else trackerA_frames_empty.wait(lock);
    }
}

void tracker_taskB(Rect &obj, queue<Mat> &frames){
    bool HOG = true;
    bool FIXEDWINDOW = false;
    bool MULTISCALE = true;
    bool LAB = true;
    bool init_tracker = true;
    KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);
    Rect result;
    Mat frame;
    int i=0;

    while(1){
        unique_lock<mutex> lock(tracker_mtx);
        if(obj.area()!=1 && !(frames.empty())){
        cout<<all_in_B.size()<<endl;
            cout<<"tracker taskkkkkkkkkkkkkkkkkkkkkkkkkkk"<<endl;
            frame = all_in_B.front();
            all_in_B.pop();
            if(init_tracker){
                tracker.init(obj, frame);
                rectangle(frame, Point(obj.x, obj.y), Point(obj.x + obj.width, obj.y + obj.height),
                      cvScalar(0, 255, 0), 2, 8);
                init_tracker=false;
            }
            else {
                result = tracker.update(frame);
                rectangle(frame, Point(result.x, result.y), Point(result.x + result.width, result.y + result.height), Scalar(0, 255, 0), 2, 8);
            }
            stringstream newstr;
            newstr<<i;
            i++;
            // if(i==30) {
            //   cout<<"o           o  ooooooooooooooooooooooooooooooooooooooooooooooo cd cajpg"<<endl;
            //   _arg.notify_all();}
            imshow("ImageB", frame);
            waitKey(1);
            cout<<"/home/d09/workspace/jjn/tt" + newstr.str() + ".jpg"<<endl;
//            imwrite( "/home/d09/workspace/jjn/tt" + newstr.str() + ".jpg", frame);
        }
        else trackerA_frames_empty.wait(lock);
    }
}


void img_assign(){
  VideoCapture captureA;
  Mat _frameA;
  captureA.open("/home/d09/workspace/jjn/2.mp4");
  VideoCapture captureB;
  // Mat _frameB;
  // captureB.open("/home/d09/workspace/jjn/1.mp4");
    /*ToDoList  如果未获得视频，则锁住当前线程（img_assign_thread）*/

  int i=0;
  while (1){
    Mat frameA;
    Mat frameB;
    captureA >> frameA;
    _frameA = frameA;
    // captureB >> frameB;
    // _frameB = frameB;
    if(0==(i%2)) {
        q1_in.push(_frameA);
    }
    else {
        q2_in.push(_frameA);}
    all_in_A.push(_frameA);
    // all_in_B.push(_frameB);
    if(!(q1_in.empty())){
          q1_in_empty.notify_all();
    }
    if(!(q2_in.empty())){
        q2_in_empty.notify_all();
    }
    cout << frameA.cols << endl;
    // cout << frameB.rows << endl;
    i++;
    cout << "img asign                    "<< i << endl;
  }
}

void img_assign2(){
  VideoCapture captureB;
  Mat _frameB;
  captureB.open("/home/d09/workspace/jjn/1.mp4");
    /*ToDoList  如果未获得视频，则锁住当前线程（img_assign_thread）*/

//  int i=0;
  while (1){
    Mat frameB;
    
    captureB >> frameB;
    _frameB = frameB;
    
    all_in_B.push(_frameB);
    }
}


/* 球机移动后需重置参数*/
/*重置完参数后再输入视频帧*/
void arg_control_task(){
  unique_lock<mutex> lock(arg_mtx);
  queue<Mat> empty0;
  queue<vector<vector<float>>> empty1;
  while(1){
    if(1){
      swap(empty0, q1_in);
      swap(empty0, q2_in);
      swap(empty0, all_in_A);
      swap(empty1, q1_out);
      swap(empty1, q2_out);
      obj_A = cvRect(-1, -1, -1, -1);
      tracker_sleep.notify_all();
      _arg.wait(lock);

//      img_assign_thread.notify_all();  //暂时觉得不需要
    }
  }
}



int main(){
    API::Set_Config("/home/d09/workspace/jjn/faster.json");
    thread detector1(detector_task1, 0, ref(q1_in), ref(q1_out));
    thread detector2(detector_task2, 0, ref(q2_in), ref(q2_out));
    thread identifier1(identifier_task1, 0);
    thread trackerA(tracker_taskA, ref(obj_A), ref(all_in_A));
    // thread trackerB(tracker_taskB, ref(obj_B), ref(all_in_B));
    thread assign(img_assign);
     thread assign2(img_assign2);

    thread query_id(identifier_task0, 0);

    thread arg_control(arg_control_task);  //参数控制

    detector1.join();
    detector2.join();
    identifier1.join();
    trackerA.join();
    // trackerB.join();
    assign.join();
     assign2.join();
    query_id.join();
    arg_control.join();
}
