#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>

#include <fann.h>
#include <fann_cpp.h>
#include <floatfann.h>


#include <ios>
#include <iostream>
#include <iomanip>
#include <fstream>




const bool isTraining  = true;
const int nTrainings = 27;
const int nImputs = 40;
const int nOutputs = 1;

int countTrainings = 0;
std::ofstream myfile;




//Callback function that simply prints the information to cout
int print_callback(FANN::neural_net &net, FANN::training_data &train,
    unsigned int max_epochs, unsigned int epochs_between_reports,
    float desired_error, unsigned int epochs, void *user_data)
{
	std::cout << "Epochs     " << std::setw(8) << epochs << ". "
         << "Current Error: " << std::left << net.get_MSE() << std::right << std::endl;
    return 0;
}

// Test function of NN in the fann C++ wrapper
void movement_test()
{
	std::cout << std::endl << "Survailance NN test started." << std::endl;

    const float learning_rate = 0.1f;
    const unsigned int num_layers = 3;
    const unsigned int num_input = nImputs;
    const unsigned int num_hidden = 6;
    const unsigned int num_output = 1;
    const float desired_error = 0.0f;
    const unsigned int max_iterations = 300000;
    const unsigned int iterations_between_reports = 1000;

	std::cout << std::endl << "Creating network." << std::endl;

    FANN::neural_net net;
    net.create_standard(num_layers, num_input, num_hidden, num_output);

    net.set_learning_rate(learning_rate);

    net.set_activation_steepness_hidden(1.0);
    net.set_activation_steepness_output(1.0);
    
    net.set_activation_function_hidden(FANN::SIGMOID_SYMMETRIC);
    net.set_activation_function_output(FANN::SIGMOID_SYMMETRIC);

    // Set additional properties such as the training algorithm
    //net.set_training_algorithm(FANN::TRAIN_QUICKPROP);

    // Output network type and parameters
	std::cout << std::endl << "Network Type                         :  ";
    switch (net.get_network_type())
    {
    case FANN::LAYER:
		std::cout << "LAYER" << std::endl;
        break;
    case FANN::SHORTCUT:
		std::cout << "SHORTCUT" << std::endl;
        break;
    default:
		std::cout << "UNKNOWN" << std::endl;
        break;
    }
    net.print_parameters();

	std::cout << std::endl << "Training network." << std::endl;

    FANN::training_data data;
    if (data.read_train_from_file("data.data"))
    {
        // Initialize and train the network with the data
        net.init_weights(data);

		std::cout << "Max Epochs " << std::setw(8) << max_iterations << ". "
            << "Desired Error: " << std::left << desired_error << std::right << std::endl;
        net.set_callback(print_callback, NULL);
        net.train_on_data(data, max_iterations,
            iterations_between_reports, desired_error);

		std::cout << std::endl << "Testing network." << std::endl;

        for (unsigned int i = 0; i < data.length_train_data(); ++i)
        {
            // Run the network on the test data
            fann_type *calc_out = net.run(data.get_input()[i]);

			std::cout << "NN test (" << std::showpos << data.get_input()[i][0] << ", " 
                 << data.get_input()[i][1] << ") -> " << *calc_out
                 << ", should be " << data.get_output()[i][0] << ", "
                 << "difference = " << std::noshowpos
                 << fann_abs(*calc_out - data.get_output()[i][0]) << std::endl;
        }

		std::cout << std::endl << "Network architecture:" << std::endl;
        net.print_connections();
        
		std::cout << std::endl << "Saving network." << std::endl;

        // Save the network in floating point and fixed point
        net.save("data_float.net");
        unsigned int decimal_point = net.save_to_fixed("data_fixed.net");
        data.save_train_to_fixed("data_fixed.data", decimal_point);

		std::cout << std::endl << "NN test completed." << std::endl;
    }
}



int main(int argc, char *argv[])
{
	
	
int iLastX = -1; 
int iLastY = -1;	

if(isTraining == true){
  myfile.open ("data.data");
  
  myfile << nTrainings << " " << nImputs << " " << nOutputs << std::endl;
}
	
cv::Mat frame;
cv::Mat back;
cv::Mat fore;
cv::VideoCapture cap("car-overhead-1.avi");
cv::BackgroundSubtractorMOG2 bg;
bg.nmixtures = 3;
bg.bShadowDetection = false;

std::vector<std::vector<cv::Point> > contours;

cv::namedWindow("Frame");
cv::namedWindow("Background");


int countIteractions = 0;
 

for(;;)
{
    cap >> frame;
    bg.operator ()(frame,fore);
    bg.getBackgroundImage(back);
    cv::erode(fore,fore,cv::Mat());
    cv::dilate(fore,fore,cv::Mat());
    cv::findContours(fore,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    cv::drawContours(frame,contours,-1,cv::Scalar(0,0,255),2);    
    
    
//Begin of position tracking        
//Calculate the moments of the thresholded image
cv::Moments oMoments = moments(fore);

double dM01 = oMoments.m01;
double dM10 = oMoments.m10;
double dArea = oMoments.m00;

//calculate the position of the target
float posX = dM10 / dArea;
float posY = dM01 / dArea;        
        
if (iLastX >= 0 && iLastY >= 0 && posX >= 0 && posY >= 0)
{
//Write to screen the coodinates of tracked target
std::cout << "PosX: " << posX << " - PosY " << posY << std::endl;


//-------------------------------------------------------
if(isTraining == true){
	countIteractions = countIteractions + 1;
	myfile << (posX/500) << " " << (posY/500) << " ";
	
	if(countIteractions == (nImputs/2)){
		countIteractions = 0;
		myfile << std::endl;
		myfile << "0.45" << std::endl;
	
		countTrainings = countTrainings + 1;
		if(countTrainings == (nTrainings)){
			std::cout << "Data training received!" << std::endl;
			myfile << std::endl;
			myfile.close();
			
			
				 try
    {
        std::ios::sync_with_stdio(); // Syncronize cout and printf output
        movement_test();
    }
    catch (...)
    {
		std::cerr << std::endl << "Abnormal exception." << std::endl;
    } 
    
    
			std::cout << "Total of detected targets: " << contours.size() << std::endl;
			exit(0);
	
		}
	}
  
}
//-------------------------------------------------------


}

iLastX = posX;
iLastY = posY;

//End of position tracking

    
    
    cv::imshow("Frame",frame);
    cv::imshow("Background",back);
    if(cv::waitKey(30) >= 0) break;
    if(!cap.read(frame)) {
		std::cout << "Video terminado!" << std::endl;
		
		if(isTraining == true){
			myfile << std::endl;
			myfile.close();
	}
	
		std::cout << "Total of detected targets: " << contours.size() << std::endl;
		exit(0);
	}
}
return 0;
}
