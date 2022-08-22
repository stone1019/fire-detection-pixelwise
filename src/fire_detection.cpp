#include <iostream>
#include "../include/fire_detection.hpp"

using namespace FD;

FireDetection::FireDetection(){
}

FireDetection::~FireDetection(){

}

void FireDetection::updateMaxI(cv::Mat &frame){
    double minVal; 
    double maxVal; 
    int minLoc; 
    int maxLoc;
    
    cv::Mat reshapedFame = frame.reshape(1); 
    
    cv::minMaxIdx(reshapedFame, &minVal, &maxVal, &minLoc, &maxLoc); 
    
    maxI = maxVal;
};

cv::Mat FireDetection::threshold(cv::Mat& maskPr, double threshold){
    cv::Mat binMask = cv::Mat::zeros(cv::Size(maskPr.cols, maskPr.rows), CV_8UC1);
    for(int col = 0; col < maskPr.cols; col++){
        for(int row = 0; row < maskPr.rows; row++){
            double pr = maskPr.at<double>(cv::Point(col, row));
            // std::cout << "pr: " << pr << std::endl;
            if(pr > threshold){
                binMask.at<int8_t>(cv::Point(col, row)) = 255;
            }
        }
    }
    return binMask;
};

cv::Mat FireDetection::findCandidatePix(cv::Mat &frame){

    FIS fis;

    TrapezoidMemFunc tramfNsYCb{-1.0, -1.0, -0.2, 0.0};
    TriangleMemFunc trimfPsYCb{0.0, 0.15, 0.3};
    TriangleMemFunc trimfPmYCb{0.2, 0.4, 0.6};
    TriangleMemFunc trimfPbYCb{0.4, 0.75, 1.1};

    TrapezoidMemFunc tramfNsCrCb{-1.0, -1.0, -0.05, 0.0};
    TriangleMemFunc trimfPsCrCb{0.0, 0.1, 0.2};
    TriangleMemFunc trimfPmCrCb{0.15, 0.4, 0.65};
    TriangleMemFunc trimfPbCrCb{0.55, 0.85, 1.15};

    TrapezoidMemFunc lowPf{0.0, 0.0, 0.05, 0.1};
    TrapezoidMemFunc mePf{0.1, 0.2, 0.3, 0.4};
    TrapezoidMemFunc hiPf{0.8, 0.9, 1.0, 1.0};

    // int img_height = frame.size().height;
    // int img_width = frame.size().width;
    // cv::Mat firePixMask = cv::Mat::zeros(cv::Size(img_width, img_height), CV_8UC3);

    // uint8_t *pixPtr = (uint8_t*)frame.data;
    // cv::Scalar_<u_int8_t> bgrPix;
    // int cn = frame.channels();

    cv::Mat yCrCbImg = cv::Mat(frame.cols, frame.rows, CV_8UC3);
    cv::cvtColor(frame, yCrCbImg, cv::COLOR_RGB2YCrCb);
    cv::Mat y, Cr, Cb;
    std::vector<cv::Mat> channel(3);
    cv::split(yCrCbImg, channel);

    y = channel[0];
    Cr = channel[1];
    Cb = channel[2];
    
    updateMaxI(yCrCbImg);
    // std::cout << "max intensity value: " << fDec.maxI << std::endl;

    cv::Mat y64f;
    cv::Mat Cr64f;
    cv::Mat Cb64f;

    y.convertTo(y64f, CV_64FC1);
    Cr.convertTo(Cr64f, CV_64FC1);
    Cb.convertTo(Cb64f, CV_64FC1);

    cv::Mat ySubCb;
    cv::subtract(y64f, Cr64f, ySubCb, cv::noArray(), CV_64F);
    cv::Mat ySubCbNorm = ySubCb / maxI;

    cv::Mat crSubCb;
    cv::subtract(Cr64f, Cb64f, crSubCb, cv::noArray(), CV_64F);
    cv::Mat crSubCbNorm = crSubCb / maxI;

    // cv::imshow("y", ySubCb);
    // cv::imshow("Cr", crSubCb);
    // cv::imshow("Cb", Cb);
    // cv::imshow("Frame", frame);

    std::vector<cv::Mat> fuzzifiedYCb = fis.fuzzify(ySubCbNorm, tramfNsYCb, trimfPsYCb, trimfPmYCb, trimfPbYCb);
    std::vector<cv::Mat> fuzzifiedCrCb = fis.fuzzify(crSubCbNorm, tramfNsCrCb, trimfPsCrCb, trimfPmCrCb, trimfPbCrCb);
    
    cv::Mat pmStack = fis.applyRule(fuzzifiedYCb, fuzzifiedCrCb);

    cv::Mat maskPr = fis.deFuzzify(pmStack, lowPf, mePf, hiPf);

    return maskPr;
}


// cv::Mat FireDetection::convertRGB2YCrCb(cv::Mat &rgbImg){
//     int imgHeight = rgbImg.size().height;
//     int imgWidth = rgbImg.size().width;
//     cv::Vec3b bgrPix;
//     cv::Vec3b yCrCb;
//     cv::Mat yCrCbImg = cv::Mat::zeros(cv::Size(imgWidth, imgHeight), CV_64FC3);

//     for(int x = 0; x < imgWidth; x++){
//         for(int y = 0; y < imgHeight; y++){
//             bgrPix = rgbImg.at<cv::Vec3b>(cv::Point(x, y));
//             yCrCb[0] = kr*bgrPix[2] + (1 - kb - kr)*bgrPix[1] + kb*bgrPix[0];
//             yCrCb[1] = 1/2 * (bgrPix[3] - yCrCb[0]) / (1 - kr);
//             yCrCb[2] = 1/2 * (bgrPix[0] - yCrCb[0] / (1 - kb));
//             yCrCbImg.at<cv::Vec3b>(cv::Point(x, y)) = yCrCb; 
//         }
//     }
//     std::cout << "--------- convert fininshed!!! -------" << std::endl;
//     return yCrCbImg;
// }

// cv::Mat FireDetection::fuzzifyYCb(cv::Mat &y, cv::Mat &Cr){
//     cv::Mat ySubCr = (y - Cr) / maxI;
//     cv::Mat degOfRelYCr = cv::Mat::zeros(cv::Size(y.cols, y.rows), CV_64FC1);
//     for(int row = 0; row < y.rows; row++){
//         for(int col = 0; col < y.cols; col++){
//             double val = ySubCr.at<double>(cv::Point(col, row));
//             // yCbMemFunc(val);
//         }
//     }
// }

// cv::Mat FireDetection::fuzzifyCrCb(cv::Mat & Cr, cv::Mat & Cb){
//     cv::Mat crSubCb = (Cr - Cb) / maxI;
//     cv::Mat defOfRelCrCb = cv::Mat::zeros(cv::Size(Cr.cols, Cr.rows), CV_64FC1);
//     for(int row = 0; row < Cr.rows; row++){
//         for(int col = 0; col < Cr.cols; col++){
//             double val = crSubCb.at<double>(cv::Point(col, row));
//         }
//     }
// }

TriangleMemFunc::TriangleMemFunc():
leftZero(0),
centralPeak(0),
rightZero(0)
{
}

TriangleMemFunc::TriangleMemFunc(double lz, double cp, double rz){
    leftZero = lz;
    centralPeak = cp;
    rightZero = rz;
}

TriangleMemFunc::~TriangleMemFunc(){
}

cv::Mat TriangleMemFunc::apply(cv::Mat& img){
    // std::cout << "trimf:" << leftZero << ", "<< centralPeak << ", " << rightZero << std::endl;
    cv::Mat Pf = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_64FC1);

    if(leftZero == 0 && rightZero ==0 && centralPeak == 0){
        return Pf;
    }

    for(int row=0; row < img.rows; row++){
        for(int col=0; col < img.cols; col++)
            {
                double pixVal = img.at<double>(cv::Point(col, row));
                // std::cout << "Pixel value: " << pixVal << std::endl;
                if(pixVal <= leftZero){
                    Pf.at<double>(cv::Point(col, row)) = 0.0;
                }
                else if(pixVal >= rightZero){
                    Pf.at<double>(cv::Point(col, row)) = 0.0;
                }
                else if(pixVal > leftZero && pixVal < centralPeak){
                    Pf.at<double>(cv::Point(col, row)) = (pixVal - leftZero) / (centralPeak - leftZero);
                    // std::cout << "between left zero and central peak: " << (pixVal - leftZero) / (centralPeak - leftZero) << std::endl;
                }
                else if(pixVal > centralPeak && pixVal < rightZero){
                    Pf.at<double>(cv::Point(col, row)) = 1 + (pixVal - centralPeak) / (centralPeak - rightZero);
                    // std::cout << "between central peak and right zero: " << 1 + (pixVal - centralPeak) / (centralPeak - rightZero) << std::endl;
                }
            }
    }
    return Pf;
}

TrapezoidMemFunc::TrapezoidMemFunc():
leftZero(0),
rightZero(0),
leftPeak(0),
rightPeak(0)
{
}

TrapezoidMemFunc::TrapezoidMemFunc(double lz, double lp, double rp, double rz){
    leftPeak = lp;
    rightPeak = rp;
    leftZero = lz;
    rightZero = rz;
}

TrapezoidMemFunc::~TrapezoidMemFunc(){}

traArea TrapezoidMemFunc::area(double th= 1.0){
    traArea tarea;
    if(th == 0){
        return tarea;
    }

    if(th == 1.0){
        tarea.area1 = (leftPeak - leftZero) / 2;
        tarea.area2 = rightPeak - leftPeak;
        tarea.area3 = (rightZero - rightPeak) / 2;

        tarea.meanX1 = leftZero + 2 * (leftPeak - leftZero) / 3;
        tarea.meanX2 = leftPeak + (rightPeak - leftPeak) / 2;
        tarea.meanX3 = rightPeak + (rightZero - rightPeak) / 3;
    }
    else{
        double leftItersect = 0;
        double rightItersect = 0;

        if(leftPeak == 0) {leftItersect = 0;}
        else
        {
            leftItersect = th * (leftPeak - leftZero) + leftZero;
        }

        if(rightPeak == 1) {rightItersect = 1;}
        else
        {
            rightItersect = (th - 1) * (rightZero - rightPeak) + rightPeak;
        }
        
        tarea.area1 = (leftItersect - leftZero) * th / 2;
        tarea.area2 = (rightItersect - leftItersect) * th;
        tarea.area3 = (rightZero - rightItersect) * th / 2;

        tarea.meanX1 = leftZero + 2 * (leftItersect - leftZero) / 3;
        tarea.meanX2 = leftItersect + (rightItersect - leftItersect) / 2; 
        tarea.meanX3 = rightItersect + (rightZero - rightItersect) / 3;
    }
    
    return tarea;
}


cv::Mat TrapezoidMemFunc::apply(cv::Mat& img){
    // std::cout << "tramf:" << leftZero << ", " << leftPeak << ", " << rightPeak << ", " << rightZero << std::endl;
    cv::Mat Pf = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_64FC1);
    if(leftZero == 0 && rightZero ==0 && leftPeak == 0 && rightPeak == 0){
        return Pf;
    }
    
    for(int col = 0; col < img.cols; col++){
        for(int row = 0; row < img.rows; row++){
            double pixVal = img.at<double>(cv::Point(col, row));
            if(pixVal <= leftZero){
                Pf.at<double>(cv::Point(col, row)) == 0.0;
            }
            else if(pixVal >= rightZero){
                Pf.at<double>(cv::Point(col, row)) == 0.0;
            }
            else if(pixVal <= rightPeak && pixVal >= leftPeak){
                Pf.at<double>(cv::Point(col, row)) == 1.0;
            }
            else if(pixVal > leftZero && pixVal < leftPeak){
                Pf.at<double>(cv::Point(col, row)) = (pixVal - leftZero) / (leftPeak - leftZero);
            }
            else if(pixVal > rightPeak && pixVal < rightZero){
                Pf.at<double>(cv::Point(col, row)) = 1 + (pixVal - rightPeak) / (rightPeak - rightZero);
            }
        }
    }
    return Pf;
}


FIS::FIS()
{
}

FIS::~FIS(){
}

std::vector<cv::Mat> FIS::fuzzify(cv::Mat& input, TrapezoidMemFunc& traNs, TriangleMemFunc& triPs, TriangleMemFunc& triPm, TriangleMemFunc& triPb){
    // cv::Mat Pf = cv::Mat::zeros(cv::Size(input.cols, input.rows), CV_64FC2);
    std::vector<cv::Mat> result, matVec;

    cv::Mat ns = traNs.apply(input);
    cv::Mat ps = triPs.apply(input);
    cv::Mat pm = triPm.apply(input);
    cv::Mat pb = triPb.apply(input);
    
  
    cv::Mat stackedPf;
    matVec.push_back(ns);
    matVec.push_back(ps);
    matVec.push_back(pm);
    matVec.push_back(pb);
    cv::merge(matVec, stackedPf);

    // std::cout << "stackedPf size: " << stackedPf.size() << std::endl;
    cv::Mat stackedPfR, sortedIdx;
    stackedPfR = stackedPf.reshape(1, (stackedPf.rows * stackedPf.cols)); 
    // std::cout << "stackedPfR size: " << stackedPfR.size() << std::endl;
    cv::sortIdx(stackedPfR, sortedIdx, cv::SORT_EVERY_ROW + cv::SORT_DESCENDING);
    // std::cout << "sorted!!!" << std::endl;
    // std::cout << "sortedIdx size before: " << sortedIdx.size() << std::endl;
    sortedIdx = sortedIdx.reshape(4, stackedPf.rows);
    result.push_back(stackedPf);
    result.push_back(sortedIdx);

    return result;
}

cv::Mat FIS::applyRule(std::vector<cv::Mat>& fuzzifiedYCb, std::vector<cv::Mat>& fuzzifiedCrCb){
    cv::Vec4i indexYCb;
    cv::Vec4d valueYCb;
    cv::Vec4i indexCrCb;
    cv::Vec4d valueCrCb;

    cv::Mat fuzzifiedYCbPr = fuzzifiedYCb[0];
    cv::Mat fuzzifiedYCbIdx = fuzzifiedYCb[1];
    cv::Mat fuzzifiedCrCbPr = fuzzifiedCrCb[0];
    cv::Mat fuzzifiedCrCbIdx = fuzzifiedCrCb[1];
    
    cv::Mat pmHi = cv::Mat::zeros(cv::Size(fuzzifiedCrCbPr.cols, fuzzifiedCrCbPr.rows), CV_64FC1);
    cv::Mat pmMe = cv::Mat::zeros(cv::Size(fuzzifiedCrCbPr.cols, fuzzifiedCrCbPr.rows), CV_64FC1);
    cv::Mat pmLo = cv::Mat::zeros(cv::Size(fuzzifiedCrCbPr.cols, fuzzifiedCrCbPr.rows), CV_64FC1);
    // std::cout << "here" << std::endl;

    // perform and operation
    for(int col =0; col < fuzzifiedCrCbPr.cols; col++){
        for(int row=0; row < fuzzifiedCrCbPr.rows; row++){
            indexYCb = fuzzifiedYCbIdx.at<cv::Vec4i>(cv::Point(col, row));
            indexCrCb = fuzzifiedCrCbIdx.at<cv::Vec4i>(cv::Point(col, row));
            valueYCb = fuzzifiedYCbPr.at<cv::Vec4d>(cv::Point(col, row));
            valueCrCb = fuzzifiedCrCbPr.at<cv::Vec4d>(cv::Point(col, row));

            int maxIdxYCb = indexYCb[0];
            int maxIdxCrCb = indexCrCb[0];
            double Pr = max(valueYCb[maxIdxYCb], valueCrCb[maxIdxCrCb]);

            // std::cout << "maxIdxYCb: " << maxIdxCrCb << ", " 
            //           << "maxIdxCrCb: " << maxIdxCrCb << ", "
            //           << "Pr: " << Pr << std::endl;

            if(maxIdxYCb == 1 && maxIdxCrCb == 2){
                pmMe.at<double>(cv::Point(col, row)) = Pr;
            }
            else if(maxIdxYCb == 1 && maxIdxCrCb == 3){
                pmMe.at<double>(cv::Point(col, row)) = Pr;
            }
            else if(maxIdxYCb == 2 && maxIdxCrCb == 1){
                pmMe.at<double>(cv::Point(col, row)) = Pr; 
            }
            else if(maxIdxYCb == 2 && maxIdxCrCb == 2){
                pmMe.at<double>(cv::Point(col, row)) = Pr;
            }
            else if(maxIdxYCb == 2 && maxIdxCrCb == 3){
                pmHi.at<double>(cv::Point(col, row)) = Pr; 
            }
            else if(maxIdxYCb == 3 && maxIdxCrCb == 1){
                pmMe.at<double>(cv::Point(col, row)) = Pr; 
            }
            else if(maxIdxYCb == 3 && maxIdxCrCb == 2){
                pmHi.at<double>(cv::Point(col, row)) = Pr; 
            }
            else if(maxIdxYCb == 3 && maxIdxCrCb == 3){
                pmHi.at<double>(cv::Point(col, row)) = Pr; 
            }
            else{
                pmLo.at<double>(cv::Point(col, row)) = Pr;
            }
        }
    }

    cv::Mat stackedPm = cv::Mat(cv::Size(fuzzifiedCrCbPr.cols, fuzzifiedCrCbPr.rows), CV_64FC3);
    std::vector<cv::Mat> stackVec;
    stackVec.push_back(pmLo);
    stackVec.push_back(pmMe);
    stackVec.push_back(pmHi);
    cv::merge(stackVec, stackedPm);

    return stackedPm;
}


cv::Mat FIS::deFuzzify(cv::Mat& input, TrapezoidMemFunc& lowMf, TrapezoidMemFunc& meMf, TrapezoidMemFunc& hiMf){
    cv::Mat fireMaskPr = cv::Mat::zeros(cv::Size(input.cols, input.rows), CV_64FC1);

    cv::Mat loPm = cv::Mat(cv::Size(input.cols, input.rows), CV_64FC1);
    cv::Mat mePm = cv::Mat(cv::Size(input.cols, input.rows), CV_64FC1);
    cv::Mat hiPm = cv::Mat(cv::Size(input.cols, input.rows), CV_64FC1);

    std::vector<cv::Mat> channel(3);
    cv::split(input, channel);

    loPm = channel[0];
    mePm = channel[1];
    hiPm = channel[2];
    
    std::vector<double> prVal;

    for(int col = 0; col < input.cols; col++){
        for(int row = 0; row < input.rows; row++){
            double lPr = loPm.at<double>(cv::Point(col, row));
            double mePr = mePm.at<double>(cv::Point(col, row));
            double hiPr = hiPm.at<double>(cv::Point(col, row));

            // std::cout << "pr1: " << lPr << ", " << "pr2: " << mePr << ", " << "pr3: " << hiPr << std::endl;
            fireMaskPr.at<double>(cv::Point(col, row)) = centerOfArea(lPr, mePr, hiPr, lowMf, meMf, hiMf);
        }
    }
    return fireMaskPr;
}

double FIS::centerOfArea(double& prob1, double& prob2, double& prob3, FD::TrapezoidMemFunc& lowMf, TrapezoidMemFunc& meMf, TrapezoidMemFunc& hiMf){

    traArea lowArea = lowMf.area(prob1);
    traArea meArea = meMf.area(prob2);
    traArea hiArea = hiMf.area(prob3);
    // std::cout << "prob1: " << prob1 << ", " << "prob2: " << prob2 << ", " << "prob3: " << prob3 << std::endl;

    double area = lowArea.area1 + lowArea.area2 + lowArea.area3 +
                  meArea.area1 + meArea.area2 + meArea.area3 +
                  hiArea.area1 + hiArea.area2 + hiArea.area3;

    double xMulArea = lowArea.area1 * lowArea.meanX1 + lowArea.area2 * lowArea.meanX2 + lowArea.area3 * lowArea.meanX3 +
                    meArea.area1 * meArea.meanX1 + meArea.area2 * meArea.meanX2 + meArea.area3 * meArea.meanX3 +
                    hiArea.area1 * hiArea.meanX1 + hiArea.area2 * hiArea.meanX2 + hiArea.area3 * hiArea.meanX3;

    double central;

    if(area != 0){
         central = xMulArea / area;
    }
    else
    {
        central = 0;
    }
    // std::cout << "central: " << central << std::endl;
    return central;
}

double FIS::min(double& fVal, double& sVal){
    if(fVal < sVal){
        return fVal;
    }
    else
    {
        return sVal;
    }   
}

double FIS::max(double& fVal, double& sVal){
    if(fVal > sVal){
        return fVal;
    }
    else
    {
        return sVal;
    }
}