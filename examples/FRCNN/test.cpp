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

mutex mtx;
mutex in1_mtx;
mutex in2_mtx;
//std::mutex mtx2;
condition_variable q1_empty;
condition_variable q2_empty;
condition_variable q1_in_empty;
condition_variable q2_in_empty;

queue<Mat> all_in;
queue<Mat> q1_in;
queue<Mat> q2_in;
queue<vector<vector<float>>> q1_out;
queue<vector<vector<float>>> q2_out;
vector<Mat> query_img;
vector<vector<float> > query_feature;
//tracker
Rect obj_A(-1,-1,-1,-1);
Rect obj_B(-1,-1,-1,-1);
queue<Mat> trackerA_frames;
condition_variable trackerA_frames_empty;
mutex tracker_mtx;

//id
condition_variable query_empty;
mutex id_mtx;
bool get_query_img = false;

Size input_geometry_ = cv::Size(224, 224);

struct q_repository{
    queue<Mat>q1_in;
    queue<vector<vector<float>>>q1_out;
    queue<Mat>q2_in;
    queue<vector<vector<float>>>q2_out;
    std::mutex mtx1;
    std::mutex mtx2;
    std::condition_variable q1_empty;
    std::condition_variable q2_empty;
} q;

inline std::string INT(float x) { char A[100]; sprintf(A,"%.1f",x); return std::string(A);};
inline std::string FloatToString(float x) { char A[100]; sprintf(A,"%.4f",x); return std::string(A);};


void identifier_task0(int gpu_id){
    // 如果接收到query_img 设置一个标志位(get_query_img) 让框画在画面上  此处保留一默认query_img
    Mat query_image = imread("/home/priv-lab1/workspace/faster_exp/team2/faster-2fc/vehicle/r.png");
    query_img.push_back(query_image);
    API::Identifier identifier(gpu_id);
    while(1){
        identifier.predict(query_img, query_feature);
        query_empty.notify_all();
        cout << "id_task0" << endl;
        if(get_query_img){
        break;
        }
    }
}

void identifier_task1(int gpu_id){
    API::Identifier identifier(gpu_id);
    int i=0;
    vector<vector<float>> targets;
    vector<float> target;
    Mat cv_image;
    bool tracker_init = true;
    while(1){
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
        cv_image = all_in.front();
        vector<Mat> gallery_imgs;
        Mat gallery_img;
        for(int j=0; j<targets.size();j++){
            target = targets[j];
            gallery_img = cv_image(Range(int(target[1]), int(target[3])), Range(int(target[0]), int(target[2])));
//            cout << int(target[1]) << " " << int(target[3]) << " " <<int(target[0]) << " " <<int(target[2]) << " " <<endl;
            gallery_imgs.push_back(gallery_img);
        }
        vector<vector<float> > gallery_features;
        identifier.predict(gallery_imgs, gallery_features);
        vector<float> sims;
        float sim_thresh=0;
        cout << query_feature[0].size() << endl;
        unique_lock <mutex> query_lck(id_mtx);
        cout << "query_feature size" << " " << query_feature.size() << endl;
        if(!query_feature.size()) query_empty.wait(query_lck);
        identifier.cosine_similarity(query_feature, gallery_features, sims);
//        std::cout<< sims.size()<<" " <<gallery_features.size()<< std::endl;
        cout << "consumer" << endl;
        float obj=0;
        int obj_index=-1;
        for(int i=0; i<sims.size(); i++){
            cout<< sims[i]<< endl;
            if ((sims[i]>sim_thresh)&&(sims[i]>obj)){
                obj = sims[i];
                obj_index = i;
            }
        }
        if(obj_index>0 && tracker_init){
            int x = int(targets[obj_index][0]);
            int y = int(targets[obj_index][1]);
            int height = int(targets[obj_index][3] - targets[obj_index][1]);
            int width = int(targets[obj_index][2] - targets[obj_index][0]);
            obj_A = cvRect(x, y, width, height);
            trackerA_frames = all_in;                                      //tracker frames  error
            tracker_init = false;
            trackerA_frames_empty.notify_all();                             //query feature 更新  得到query_img后 设置get_query_img开始更新query_feature
        }
        all_in.pop();
   }
}


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
//          imwrite( "/home/priv-lab1/workspace/faster_exp/team2/faster-2fc/" + newstr.str() + ".jpg", cv_image );
          vector<caffe::Frcnn::BBox<float> > results;
          boost::shared_ptr<Blob<float> > img_crop;
          detector.predict(cv_image, results, img_crop);
          Mat crop_img;
          vector<Mat> crop_imgs;
          cout << results.size() << endl;
          for (size_t obj = 0; obj < results.size(); obj++) {
//            std::cout << "ok" << results[obj].id << "  " << INT(results[obj][0]) << " " << INT(results[obj][1]) << " " << INT(results[obj][2]) << " " << INT(results[obj][3]) << "     " << FloatToString(results[obj].confidence) << std::endl;
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
//          imwrite( "/home/priv-lab1/workspace/faster_exp/team2/faster-2fc/" + newstr.str() + ".jpg", cv_image );
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

void img_assign(queue<Mat> &_q1_in, queue<Mat> &_q2_in, queue<Mat> &_all_in){
    VideoCapture capture;
    Mat frame;
    Mat temp;
    capture.open("/home/priv-lab1/workspace/faster_exp/team2/faster-2fc/1.mp4");
    int i=0;
    while (1){
        capture >> temp;
        frame = temp;
//        if(frame.cols==0) break;
        if(0==(i%2)) {
            _q1_in.push(frame);
            _all_in.push(frame);}
        else {
            _q2_in.push(frame);
            _all_in.push(frame);}
        if(!(q1_in.empty())){
            q1_in_empty.notify_all();
        }
        if(!(q2_in.empty())){
            q2_in_empty.notify_all();
        }
        i++;
        if(i==10) break;
        cout << _q1_in.size() << "q1" << endl;
        cout << _q2_in.size() << "q2" << endl;
     }
}

// tracker 初始化条件 锁条件 帧存储
void tracker_task(Rect &obj, queue<Mat> &frames){
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
        if(obj_A.area()!=1 && !(all_in.empty())){
        cout<<all_in.size()<<endl;
            cout<<"tracker taskkkkkkkkkkkkkkkkkkkkkkkkkkk"<<endl;
            frame = all_in.front();
            frames.pop();
            if(init_tracker){
                tracker.init(obj, frame);
                rectangle(frame, Point(obj.x, obj.y), Point(obj.x + obj.width, obj.y + obj.height),
                      cvScalar(0, 255, 0), 2, 8);
                init_tracker=false;
            }
            else {
                result = tracker.update(frame);
                BoundRect_save.insert(BoundRect_save.end(), result);
                rectangle(frame, Point(result.x, result.y), Point(result.x + result.width, result.y + result.height), Scalar(0, 255, 0), 2, 8);
            }
            stringstream newstr;
            newstr<<i;
            i++;
//            imshow("Image", frame);
            cout<<"/home/priv-lab1/workspace/faster_exp/team2/faster-2fc/tt" + newstr.str() + ".jpg"<<endl;
            imwrite( "/home/priv-lab1/workspace/faster_exp/team2/faster-2fc/tt" + newstr.str() + ".jpg", frame);
        }
        else trackerA_frames_empty.wait(lock);
    }
}


int main()
{
    API::Set_Config("/home/priv-lab1/workspace/faster_exp/team2/faster-2fc/faster.json");
    thread detector1(detector_task1, 1, ref(q1_in), ref(q1_out));
    thread detector2(detector_task2, 2, ref(q2_in), ref(q2_out));
    thread identifier1(identifier_task1, 0);
    thread trackerA(tracker_task, ref(obj_A), ref(trackerA_frames));
//    thread trackerB(tracker_task);
    thread assign(img_assign, ref(q1_in), ref(q2_in), ref(all_in));

    thread query_id(identifier_task0, 0);

    detector1.join();
    detector2.join();
    identifier1.join();
    trackerA.join();
    assign.join();
    query_id.join();
}