#include <iostream>
#include<opencv2/opencv.hpp>
#include <omp.h>

using namespace std;
using namespace cv;

bool processInput(std::vector<cv::Mat> imgs)
{
    const int inputC = 3;
    const int inputH = 1080;
    const int inputW = 1920;

    // Fill data buffer
    float* hostDataBuffer = new float[imgs.size()*3*1920*1080];
    // Host memory for input buffer
    for (int i = 0, volImg = inputC * inputH * inputW; i < imgs.size(); ++i)
    {
        for (int c = 0; c < inputC; ++c)
        {
            for (unsigned j = 0, volChl = inputH * inputW; j < volChl; ++j)
            {
                hostDataBuffer[i * volImg + c * volChl + j] = float(imgs[i].data[j * inputC + 2 - c])/255.;
            }
        }
    }
    return true;
}


int main()
{
    int size = 16;
    double sum=0;
    Mat img=imread("/home/jiangzhiqi/1.jpg");
    std::vector<cv::Mat> imgs;
    imgs.push_back(img);
    imgs.push_back(img);
    for(int j=0;j<1000;j++)
    {
        double time11=omp_get_wtime()*1000;
        #pragma omp parallel for num_threads(2) schedule(static, 8)
        for (int i = 0; i < size; ++i)
        {
            processInput(imgs);
        }
        double time22=omp_get_wtime()*1000;
        sum=sum+time22-time11;
        cout<<time22-time11<<"ms"<<endl;
    }
    cout<<"average="<<sum/1000<<endl;
    return 0;
}