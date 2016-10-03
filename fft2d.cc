// Distributed two-dimensional Discrete FFT transform
// Rohit Ganesan
// ECE 4122


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <signal.h>
#include <math.h>
#include <mpi.h>

#include "Complex.h"
#include "InputImage.h"

using namespace std;

void Transform1D(Complex* h, int w, Complex* H);
void add(Complex *input, int w, Complex *output);
void inverse2D(Complex* H, int w);
void deliver(Complex* result, int rank, int numTasks, int w, int h, InputImage img);

void Transform2D(const char* inputFN)
{ // Do the 2D transform here.

  // 1) Use the InputImage object to read in the Tower.txt file and
  //    find the width/height of the input image.
  // 2) Use MPI to find how many CPUs in total, and which one
  //    this process is
  // 3) Allocate an array of Complex object of sufficient size to
  //    hold the 2d DFT results (size is width * height)
  // 4) Obtain a pointer to the Complex 1d array of input data
  // 5) Do the individual 1D transforms on the rows assigned to your CPU
  // 6) Send the resultant transformed values to the appropriate
  //    other processors for the next phase.
  // 6a) To send and receive columns, you might need a separate
  //     Complex array of the correct size.
  // 7) Receive messages from other processes to collect your columns
  // 8) When all columns received, do the 1D transforms on the columns
  // 9) Send final answers to CPU 0 (unless you are CPU 0)
  //   9a) If you are CPU 0, collect all values from other processors
  //       and print out with SaveImageData().

  InputImage image (inputFN);  // Create the helper object for reading the image
  int imgWidth = image.GetWidth();
  int imgHeight = image.GetHeight();
  int numTasks, rank; // Total number of ranks, computer rank
  //get MPI info
  MPI_Comm_size(MPI_COMM_WORLD,&numTasks);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  Complex* after1DResult = new Complex[imgWidth*imgHeight];
  Complex* after1DResult2 = new Complex[imgWidth*imgHeight];
  Complex* data = image.GetImageData(); //Image Data


  //do transform 1D
  Transform1D(data, imgWidth,after1DResult);
  deliver(after1DResult, rank, numTasks, imgWidth, imgHeight,image); //Push
  //do transform 2D
  Transform1D(after1DResult,imgWidth, after1DResult2);
  deliver(after1DResult2, rank, numTasks, imgWidth,imgHeight, image);//Push

//delete arrays
delete [] after1DResult; delete [] after1DResult2;


}

void deliver(Complex* result, int rank, int numTasks, int w, int h, InputImage image){
  //Complex* input = new Complex(w*w);
  int rc;
  if (rank != 0) { //if CPU is not 0
    rc = MPI_Send(result, w*h*sizeof(Complex), MPI_CHAR, 0, 0, MPI_COMM_WORLD); //send
    MPI_Status status;//MPI status
    rc = MPI_Recv(result, w*h*sizeof(Complex), MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status); //recieve

  } else { //every CPU other than 0
    int rowsPerRank = h / numTasks;
    for (int i = 1; i < numTasks; i++) {
      Complex* input = new Complex[w*h];
      MPI_Status status;
      rc = MPI_Recv(input, w*h*sizeof(Complex), MPI_CHAR, i, 0, MPI_COMM_WORLD, &status);

      int startIndex = w*i*rowsPerRank;
      for (int k = 0; k <rowsPerRank; k++){
        for(int i = 0; i < w; i++) {
        result[startIndex+i+(k*w)] = input[startIndex+i+(k*w)];
        }
      }
      
      delete [] input;
    }
    inverse2D(result,w);
    
    image.SaveImageData("MyAfter2D.txt", result, w, h);
    for (int r = 1; r < numTasks; r++) {
      rc = MPI_Send(result, w*h*sizeof(Complex), MPI_CHAR, r,0,MPI_COMM_WORLD);
    }

  }
  
}
//adding the current array to previous array
void add(Complex *input, int w, Complex *output){
  int numTasks, rank;
  MPI_Comm_size(MPI_COMM_WORLD,&numTasks);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  int rowsPerRank = w/numTasks;
  int start = rank*rowsPerRank*w;

  for(int i = 0; i < rowsPerRank; i++){
    for (int j = 0; j < w; i++){
      output[start+j+i*w] = input[start+j+i*w];
    }
  }
}
//transpose operation
void inverse2D(Complex* H, int w){
  Complex* comp = new Complex[w*w];
  for (int i = 0; i < w; i++) {
    for (int j = 0; j < w; j++) {
      comp[(j*w)+i] = H[(w*i)+j];
    }
  }
  for (int k = 0; k < (w*w); k++) {
    H[k] = comp[k];
  }
  delete [] comp;
}

void Transform1D(Complex* h, int w, Complex* H)
{
  // Implement a simple 1-d DFT using the double summation equation
  // given in the assignment handout.  h is the time-domain input
  // data, w is the width (N), and H is the output array.
  int numTasks,rank;
  MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  Complex value;
  int rowsPerRank = w/numTasks;
  int index = rowsPerRank*w*rank;
  

  //loop for 1D FFT
  /*
  int i = 0; int j = 0; int k = 0;
  while(i < rowsPerRank) {
    while(j < w){
      while(k < w){
        double cosTerm = cos((2*M_PI*j*k)/w);
        double sinTerm = -sin((2*M_PI*j*k)/w);
        Complex term(cosTerm,sinTerm);
        term = term*h[(i*w)+index+k];
        value = value+term;
        k++;
      }
      H[index+j+(i*w)] = value;
      value = Complex(0.0,0.0);
      j++;
    }
    i++;
  }*/
  for(int i = 0; i < rowsPerRank; i++) {
    for(int j = 0; j < w; j++){
      for(int k = 0; k < w; k++){
        double cosTerm = cos((2*M_PI*j*k)/w);
        double sinTerm = -sin((2*M_PI*j*k)/w);
        Complex term(cosTerm,sinTerm);
        term = term*h[(i*w)+index+k];
        value = value+term;

      }
      H[index+j+(i*w)] = value;
      value = Complex(0.0,0.0);
    }
  }

}



int main(int argc, char** argv)
{
  
  string fn("Tower.txt"); // default file name
  if (argc > 1) fn = string(argv[1]);  // if name specified on cmd line
  // MPI initialization here
  MPI_Init(&argc, &argv);
  Transform2D(fn.c_str()); // Perform the transform.
  MPI_Finalize();// Finalize MPI here
}