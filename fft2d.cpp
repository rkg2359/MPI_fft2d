// Distributed two-dimensional Discrete FFT transform
// Mac Clayton, 2012
// ECE4893 Project 1

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


// Function to perform the 1D FFT transform
void Transform1D(Complex* h, int w, Complex* H)
{   
    // Complex variables:
    Complex weight;
    Complex sum;
    
    // Simple FFT calculations:
    for(int n = 0; n < w; ++n)
    {
        sum = Complex(0,0);
        for( int k = 0; k < w; ++k)
        {
            weight = Complex( cos(2 * M_PI * n * k / w), -sin(2*M_PI * n * k / w));
            sum = sum + weight * h[k];
        }
        H[n] = sum;
    }
} // End Transform1D



// Function to perform the 2D transform (with MPI)
void Transform2D(const char* inputFN) 
{
    InputImage image(inputFN);  // Create the helper object for reading the image
    // Step (1) in the comments is the line above.
    // Your code here, steps 2-9
    int numtasks;                           // Total number of tasks
    int rank;                               // Computer rank
    int imgHeight, imgWidth;                // Dimensions of the image
    Complex *imgdata, *H;                   // Origninal data and after 1D transform
    int start_row;                          // Starting row
    int i,r,c;                              // Loop variables
    int rc;                                 // MPI return value
    int tgt, source;                        // Read/Write buffer indices
    int sender;                             // Who sent a message
    double *recvBuff;                       // Receive data buffer
    double *sendBuff;                       // Send data buffer
    int message_length;                     // Length of send/receive messages
    MPI_Status message_status;              // Status of send/receive messages
    MPI_Request *Rrequests, *Srequests;     // Array of receive/send requests

    // get image dimensions and data
    imgHeight = image.GetHeight();          // Image Height
    imgWidth = image.GetWidth();            // Image width
    imgdata = image.GetImageData();         // Image Data
  
    // get MPI info
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 
    // allocate space for H
    H = new Complex[(imgHeight / numtasks) * imgWidth];
 
    // Do the 1D transform
    start_row = (imgHeight / numtasks) * rank;
    for (i = 0; i < imgHeight / numtasks; ++i)
    {
        Transform1D(imgdata + (imgWidth * (i + start_row)), imgWidth, H + imgWidth*i);
    }
  
    // Create send buffers, need to be 1 row for each task, with (Height * Width)/numtasks elements per message
    // Not using an array of Complex, but instead splitting the real and complex parts up (hence the 2 and +=2 that occurs later)
    sendBuff = new double[2 * imgHeight * imgWidth / numtasks];
    message_length = 2 * (imgHeight / numtasks) * (imgWidth / numtasks);
    tgt = 0;

    // Fill the buffers and transpose at the same time:
    for (c = 0; c < imgWidth; ++c)
    {
        for (r = 0; r < imgHeight / numtasks; ++r)
        {
            sendBuff[tgt] = H[(r * imgWidth) + c].real;
            ++tgt;
            sendBuff[tgt] = H[(r * imgWidth) + c].imag;
            ++tgt;
        }
    }
    
    // Create receive buffer and requests
    recvBuff = new double[2 * imgHeight*imgWidth / numtasks];
    Rrequests = new MPI_Request[numtasks - 1];
    Srequests = new MPI_Request[numtasks];

    // Receive the data:
    for(i = 0; i < (numtasks - 1); ++i)
    {
        rc = MPI_Irecv(
            recvBuff + i*message_length,    // Location of receive buffer
            message_length,                 // Length of Message
            MPI_DOUBLE,                     // Data Type
            MPI_ANY_SOURCE,                 // Receive from any source
            0,                              // Tag of zero
            MPI_COMM_WORLD,                 // Communicate with all members
            Rrequests+i);                   // Requests array
    }

    // Send the necessary data:
    for(i = 0; i < numtasks; ++i)
    {
        if (i != rank) {
            rc = MPI_Isend(
                sendBuff + i*message_length,// Location of send buffer
                message_length,             // Length of Message
                MPI_DOUBLE,                 // Data Type
                i,                          // Destination rank
                0,                          // Tag of zero
                MPI_COMM_WORLD,             // Communicate with all members
                Srequests+i);               // Send array
        }
    }

    // Get the data from me to myself instead of using MPI
    source = message_length * rank;
    for(r = rank * imgHeight / numtasks; r < (rank + 1) * imgHeight / numtasks; ++r)
    {
        for(c = rank * imgWidth / numtasks; c < (rank + 1)*imgWidth / numtasks; ++c)
        {
            imgdata[(r * imgWidth) + c] = Complex(sendBuff[source], sendBuff[source + 1]);
            source += 2;
        }
    }
  
    // Read all the sent data
    for(i = 0; i < numtasks - 1; ++i)
    {
        MPI_Wait(Rrequests+i, &message_status);
        sender = message_status.MPI_SOURCE;
        source = message_length*i;
        for(r = rank * imgHeight/numtasks; r < (rank + 1) * imgHeight / numtasks; ++r )
        {
            for(c = sender * imgWidth / numtasks; c < (sender + 1) * imgWidth / numtasks; ++c)
            {
                imgdata[(r * imgWidth) + c] = Complex(recvBuff[source], recvBuff[source + 1]);
                source += 2;
            }
        }
    }

    // Do the transform:
    start_row = (imgHeight / numtasks) * rank;
    for (i = 0; i < imgHeight / numtasks; ++i)
    {
        Transform1D(imgdata + (imgWidth * (i + start_row)), imgWidth, H + imgWidth * i);
    }
  
    // free up the buffer and request memory
    delete[] recvBuff;
    delete[] sendBuff;
    delete[] Srequests;
    delete[] Rrequests;
  
    // Send all the data back to rank zero:
    message_length = 2 * imgWidth * imgHeight / numtasks;

    // If I'm rank zero:
    if (rank == 0)
    {
        // Create receive buffer and Requests
        recvBuff = new double[message_length * (numtasks)];
        Rrequests = new MPI_Request[numtasks - 1];

        // Recieve data:
        for(i = 0; i < numtasks - 1; ++i)
        {
            rc = MPI_Irecv(recvBuff + i * message_length, message_length,  MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, Rrequests + i);
        }
        // fill the data I should send to myself rather than use a message
        source = 0;
        for(c = 0; c < imgWidth / numtasks; ++c)
        {
            for(r = 0; r < imgHeight; ++r)
            {
                imgdata[r * imgWidth + c] = H[source];
                ++source;
            }
        }

        // read the recieve data
        for(i = 0; i < numtasks - 1; ++i)
        {
            MPI_Wait(Rrequests + i, &message_status);
            sender = message_status.MPI_SOURCE;
            source = message_length * i;
            for(c = sender * imgWidth / numtasks; c < (sender + 1) * imgWidth / numtasks; ++c)
            {
                for(r = 0; r < imgHeight; ++r)
                {
                    imgdata[r * imgWidth + c] = Complex(recvBuff[source], recvBuff[source + 1]);
                    source += 2;
                }
            }
        }
  
        // Delete/free data:
        delete[] Rrequests;
        delete[] recvBuff;
    }

    // If I'm not node zero:
    else
    {
        // Create send buffer:
        sendBuff = new double[message_length];
        tgt = 0;

        // Fill up the send buffer with data:
        for(i = 0; i < imgWidth * imgHeight / numtasks; ++i)
        {
            sendBuff[tgt] = H[i].real;
            ++tgt;
            sendBuff[tgt] = H[i].imag;
            ++tgt;
        }
        // Send the data:
        Srequests = new MPI_Request[0];
        rc = MPI_Isend(sendBuff, message_length, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, Srequests);
        MPI_Wait(Srequests, &message_status);

        // Delete/free data:
        delete[] sendBuff;
        delete[] Srequests;
    }

    // Delete the H array:
    delete[] H;

    // Save the image data into the file Tower=DFT2D.txt:
    if(rank == 0)
    {
        image.SaveImageData("Tower-DFT2D.txt", imgdata, imgWidth, imgHeight);
    }
} // End Transform2D


// Main Function
int main(int argc, char** argv)
{
    string fn("Tower.txt");                 // default file name
    if (argc > 1) fn = string(argv[1]);     // if name specified on cmd line
    MPI_Init(&argc, &argv);                 // Start MPI:
    Transform2D(fn.c_str());                // Perform the transform.
    MPI_Finalize();                         // Finalize MPI
} // End Main