
#ifndef FIRE_DETECTION_HPP
#define FIRE_DETECTION_HPP

#include <opencv2/opencv.hpp>

namespace FD{

struct funcType
{
    enum Type{
        TriangleFunc = 0,
        TrapezoidFunc = 1,
    };
};

typedef struct trapezoidArea{
    double area1 = 0;
    double area2 = 0;
    double area3 = 0;
    double meanX1 = 0;
    double meanX2 = 0;
    double meanX3 = 0;
} traArea;


class FireDetection{

    public:
        FireDetection();
        ~FireDetection();
        float kr = 0.0722;
        float kb = 0.221;
        double maxI = 0;
        void updateMaxI(cv::Mat &);
        // cv::Mat convertRGB2YCrCb(cv::Mat &);
        // cv::Mat fuzzifyYCb(cv::Mat &, cv::Mat &);
        // cv::Mat fuzzifyCrCb(cv::Mat &, cv::Mat &);
        cv::Mat findCandidatePix(cv::Mat &);
        cv::Mat threshold(cv::Mat&, double);
};



class TriangleMemFunc{
    public: 
        TriangleMemFunc();
        TriangleMemFunc(double, double, double);
        ~TriangleMemFunc();
        cv::Mat apply(cv::Mat&);
    protected:
        double leftZero;
        double rightZero; 
        double centralPeak;
};

class TrapezoidMemFunc{
    public:
        TrapezoidMemFunc();
        TrapezoidMemFunc(double, double, double, double);
        ~TrapezoidMemFunc();
        FD::traArea area(double);
        cv::Mat apply(cv::Mat&);
        
    protected:
        double leftZero;
        double rightZero; 
        double leftPeak;
        double rightPeak;
};

}

class FIS{
    public:
        FIS();
        ~FIS();
        std::vector<cv::Mat> fuzzify(cv::Mat&, FD::TrapezoidMemFunc&, FD::TriangleMemFunc&, FD::TriangleMemFunc&, FD::TriangleMemFunc&);
        cv::Mat applyRule(std::vector<cv::Mat>&, std::vector<cv::Mat>&);
        cv::Mat deFuzzify(cv::Mat&, FD::TrapezoidMemFunc&, FD::TrapezoidMemFunc&, FD::TrapezoidMemFunc&);
        double centerOfArea(double&, double&, double&, FD::TrapezoidMemFunc&, FD::TrapezoidMemFunc&, FD::TrapezoidMemFunc&);
        double min(double&, double&);
        double max(double&, double&);
       
};
#endif


