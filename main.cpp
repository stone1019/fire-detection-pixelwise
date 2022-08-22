#include <iostream>
#include "include/fire_detection.hpp"

using namespace FD;

int main(int argc, char** argv) {

    FireDetection fDec;
    
    if(*argv[1] == 'c'){
        cv::VideoCapture cap(argv[2]);
        if(!cap.isOpened()){
            std::cout << "camera open failed!!!" << std::endl;
            return -1;
        }
        
        cv::Mat frame, snapShot;

        // int frame_width= 480;
        // int frame_height= 544;
        // cv::VideoWriter video("video2.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 20, cv::Size(frame_width, frame_height), 0);
        // if ( !video.isOpened() ) //if not initialize the VideoWriter successfully, exit the program
        // {
        //     std::cout << "ERROR: Failed to write the video" << std::endl;
        //     return -1;
        // }

        while(cap.isOpened()){
            cap.read(frame);

            if(!frame.size().width > 0){
                break;
            }

            cv::Mat maskPr = fDec.findCandidatePix(frame);
            
            cv::Mat mask = fDec.threshold(maskPr, 0.6);

            // cv::Mat maskRgb;
            // cv::cvtColor(mask, maskRgb, cv::COLOR_GRAY2RGB);
            cv::imshow("frame", frame);
            cv::imshow("mask", mask);

            // cv::Mat frameGRAY;
            // cv::cvtColor(frame, frameGRAY, cv::COLOR_RGB2GRAY);
            // cv::Mat display_img = cv::Mat(cv::Size(frame.cols, frame.rows* 2), CV_8UC1);
            // std::cout << "frame size: " << display_img.cols << ", " << display_img.rows << ", " << display_img.channels() <<std::endl;
            // std::cout << "mask size: " << maskRgb.cols << ", " << maskRgb.rows << ", " << maskRgb.channels() << std::endl;
            // cv::vconcat(frameGRAY, mask, display_img);
            // cv::imshow("display", display_img);
            // video.write(display_img);
            cv::waitKey(10);
        }

        cap.release();
        cv::destroyAllWindows();
    }
    else if(*argv[1] == 'i'){
        cv::Mat image = cv::imread(argv[2]);

        cv::resize(image, image, cv::Size(640, 480));
        cv::Mat maskPr = fDec.findCandidatePix(image);    
        cv::Mat mask = fDec.threshold(maskPr, 0.5);
        cv::imshow("frame", image);
        cv::imshow("mask", mask);
        cv::waitKey(0);
    }

    else if(*argv[1]=='v'){
        cv::VideoCapture cap(argv[2]);
        
        if(cap.isOpened() == false){
            std::cout << "video file open failed !!!" <<std::endl; 
        }
        
        while(cap.isOpened()){
            cv::Mat frame;
            cap >> frame;
            cv::resize(frame, frame, cv::Size(640, 480));
            cv::Mat maskPr = fDec.findCandidatePix(frame);
            
            cv::Mat mask = fDec.threshold(maskPr, 0.5);
        
            cv::imshow("frame", frame);
            cv::imshow("mask", mask);
            cv::waitKey(1);
        }
    }
    return 0;
}
