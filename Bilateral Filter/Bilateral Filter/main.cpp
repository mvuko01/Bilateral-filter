
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
using namespace cv;
using namespace std::chrono;

float distance(int x, int y, int i, int j) {
    return float(sqrt(pow(x - i, 2) + pow(y - j, 2)));
}

double gaussian(float x, double sigma) {
    return exp(-(pow(x, 2)) / (2 * pow(sigma, 2))) / (2 * CV_PI * pow(sigma, 2));
}

void applyBilateralFilter(const Mat& source, Mat& filteredImage, int x, int y, int diameter, double sigmaI, double sigmaS) {
    
    for (int channel = 0; channel < 3; channel++) {
		double iFiltered = 0;
		double wP = 0;
		int neighbor_x = 0;
		int neighbor_y = 0;
		int half = diameter / 2;
        double gi = 0;
        double gs = 0;
        double w = 0;

        for (int i = 0; i < diameter; i++) {
            for (int j = 0; j < diameter; j++) {
				neighbor_x = x - (half - i);
				neighbor_y = y - (half - j);
                if (neighbor_x >= source.rows) {
					neighbor_x -= source.rows;
				}
                else if (neighbor_x < 0) {
					neighbor_x += source.rows;
				}
                if (neighbor_y >= source.cols) {
					neighbor_y -= source.cols;
				}
                else if (neighbor_y < 0) {
					neighbor_y += source.cols;
				}
				gi = gaussian(source.at<Vec3b>(neighbor_x, neighbor_y)[channel] - source.at<Vec3b>(x, y)[channel], sigmaI);
				gs = gaussian(distance(x, y, neighbor_x, neighbor_y), sigmaS);
				w = gi * gs;
				iFiltered += source.at<Vec3b>(neighbor_x, neighbor_y)[channel] * w;
				wP += w;
			}
		}
		iFiltered /= wP;
		filteredImage.at<Vec3b>(x, y)[channel] = static_cast<uchar>(iFiltered);
	}
}

Mat bilateralFilterOwn(const Mat& source, int diameter, double sigmaI, double sigmaS) {
    Mat filteredImage = Mat::zeros(source.size(), source.type());
    int width = source.cols;
    int height = source.rows;

    for (int i = 2; i < height - 2; i++) {
        for (int j = 2; j < width - 2; j++) {
            applyBilateralFilter(source, filteredImage, i, j, diameter, sigmaI, sigmaS);
        }
    }

    return filteredImage;
}



int main(int argc, char** argv) {
    Mat src;
    src = imread("C:/Users/Korisnik/Desktop/NAR/noise.jpg");

    if (!src.data)
    {
        printf("No image data \n");
        return -1;
    }

    Mat filteredImageOpenCV;
    bilateralFilter(src, filteredImageOpenCV, 8, 20.0, 20.0);

    auto start = high_resolution_clock::now();

    Mat filteredImageOwn = bilateralFilterOwn(src, 8, 20.0, 20.0);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    std::cout << "Time of execution on GPU is: " << duration.count() << " ms" << std::endl;

    printf("Filtered image data: %d\n", filteredImageOwn.data);
    
    imwrite("C:/Users/Korisnik/Desktop/NAR/filtered_image_OpenCV.jpg", filteredImageOpenCV);
    imwrite("C:/Users/Korisnik/Desktop/NAR/filtered_image_own.jpg", filteredImageOwn);
    waitKey(0);

    return 0;
}