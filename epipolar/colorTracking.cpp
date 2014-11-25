#include <vector>
#include <cv.h>
#include <cvaux.h>
#include <highgui.h>
#include <iostream>
#include <list>

#include <opencv2/video/background_segm.hpp>
#include <opencv2/legacy/blobtrack.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>

#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>

#include <math.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <termios.h>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    VideoCapture cap1(1); //capture the video from webcam
    VideoCapture cap2(2); //capture the video from webcam

    if ( !cap1.isOpened() )  // if not success, exit program
    {
        cout << "Cannot open the web cam" << endl;
        return -1;
    }

    if ( !cap2.isOpened() )  // if not success, exit program
    {
        cout << "Cannot open the web cam" << endl;
        return -1;
    }

    //=============== OBJECT CONTROL ==============================================//
    namedWindow("Object", CV_WINDOW_AUTOSIZE); //create a window called "Object"

    /*
    // Camera 1
    int iLowH = 0;
    int iHighH = 50;

    int iLowS = 90;
    int iHighS = 255;

    int iLowV = 130;
    int iHighV = 255;
	*/

    //Camera 2
    int iLowH = 0;
    int iHighH = 10;

    int iLowS = 130;
    int iHighS = 255;

    int iLowV = 140;
    int iHighV = 255;
	

    //Create trackbars in "Object" window
    createTrackbar("LowH", "Object", &iLowH, 179); //Hue (0 - 179)
    createTrackbar("HighH", "Object", &iHighH, 179);

    createTrackbar("LowS", "Object", &iLowS, 255); //Saturation (0 - 255)
    createTrackbar("HighS", "Object", &iHighS, 255);

    createTrackbar("LowV", "Object", &iLowV, 255);//Value (0 - 255)
    createTrackbar("HighV", "Object", &iHighV, 255);
    //=============== object control ==============================================//

    //Capture a temporary image from the camera
    Mat imgTmp1;
    Mat imgTmp2;
    cap1.read(imgTmp1);
    cap2.read(imgTmp2);
    

    while (true)
    {
        Mat imgOriginal1;
        bool bSuccess1 = cap1.read(imgOriginal1); // read a new frame from video
        Mat imgOriginalCopy1; // make copy of imgOriginal
        cap1.read(imgOriginalCopy1);

        Mat imgOriginal2;
        bool bSuccess2 = cap2.read(imgOriginal2); // read a new frame from video
        Mat imgOriginalCopy2; // make copy of imgOriginal
        cap2.read(imgOriginalCopy2);

        if (!bSuccess1) //if not success, break loop
        {
            cout << "Cannot read a frame from video stream" << endl;
            break;
        }

        if (!bSuccess2) //if not success, break loop
        {
            cout << "Cannot read a frame from video stream" << endl;
            break;
        }

        //==================== OBJECT DETECTION CAM1 ===========================================================================//
        Mat imgHSV1;
        cvtColor(imgOriginal1, imgHSV1, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

        Mat imgThresholded1;
        inRange(imgHSV1, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded1); //Threshold the image

        //morphological opening (removes small objects from the foreground)
        erode(imgThresholded1, imgThresholded1, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
        dilate(imgThresholded1, imgThresholded1, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

        //morphological closing (removes small holes from the foreground)
        dilate(imgThresholded1, imgThresholded1, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
        erode(imgThresholded1, imgThresholded1, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

        //Calculate the moments of the thresholded image
        Moments oMoments1 = moments(imgThresholded1);

        double dM011 = oMoments1.m01;
        double dM101 = oMoments1.m10;
        double dArea1 = oMoments1.m00;
        int posX1, posY1;

        // if the area <= 10000, I consider that the there are no object in the image
        //and it's because of the noise, the area is not zero
        if (dArea1 > 10000)
        {
            //calculate the position of the ball
            posX1 = dM101 / dArea1;
            posY1 = dM011 / dArea1;

            // Draw a circle
            circle( imgOriginalCopy1, Point(posX1,posY1), 16.0, Scalar( 0, 0, 255), 3, 8 );

            cout << "Cam 1 position: \t";
            cout << posX1 << "\t";
            cout << posY1 << "\n";
        }
        imshow("Thresholded Image 1", imgThresholded1); //show the thresholded image
        //==================== object detection ===========================================================================//

        //==================== OBJECT DETECTION CAM2 ===========================================================================//
        Mat imgHSV2;
        cvtColor(imgOriginal2, imgHSV2, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

        Mat imgThresholded2;
        inRange(imgHSV2, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded2); //Threshold the image

        //morphological opening (removes small objects from the foreground)
        erode(imgThresholded2, imgThresholded2, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
        dilate(imgThresholded2, imgThresholded2, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

        //morphological closing (removes small holes from the foreground)
        dilate(imgThresholded2, imgThresholded2, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
        erode(imgThresholded2, imgThresholded2, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

        //Calculate the moments of the thresholded image
        Moments oMoments2 = moments(imgThresholded2);

        double dM012 = oMoments2.m01;
        double dM102 = oMoments2.m10;
        double dArea2 = oMoments2.m00;
        int posX2, posY2;

        // if the area <= 10000, I consider that the there are no object in the image
        //and it's because of the noise, the area is not zero
        if (dArea2 > 10000)
        {
            //calculate the position of the ball
            posX2 = dM102 / dArea2;
            posY2 = dM012 / dArea2;

            // Draw a circle
            circle( imgOriginalCopy2, Point(posX2,posY2), 16.0, Scalar( 0, 0, 255), 3, 8 );
    
    		cout << "Cam 2 position: \t";
            cout << posX2 << "\t";
            cout << posY2 << "\n\n";
        }
        imshow("Thresholded Image 2", imgThresholded2); //show the thresholded image
        //==================== object detection ===========================================================================//

        imshow("Original 1", imgOriginal1); //show the original image
        imshow("Original 2", imgOriginal2); //show the original image
	

            if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
            {
            cout << "esc key is pressed by user" << endl;
                    break;
            }
    }
    return 0;
}
