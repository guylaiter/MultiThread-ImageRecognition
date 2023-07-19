#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "helper.h"

int main(int argc, char *argv[])
{
    int rank, size;
    int numberOfPictures, numberOfObjects;
    double matchingThreshold;

    int picturesFinished = 0;
    int picturesInProgress = 0;
    
    Picture *pictures;
    Object *objects;
    MPI_Status status;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check if number of processes is greater than 2
    if (size < 2)
    {
        printf("Number of processes must be 2 or more \n");
        MPI_Finalize();
        return 0;
    }

    double timer = MPI_Wtime();

    // Read input files
    if (rank == 0)
        readInputFile(INPUT_FILE, &pictures, &objects, &matchingThreshold, &numberOfPictures, &numberOfObjects);

    // Broadcast matching threshold, number of pictures, number of objects to all processes
    MPI_Bcast(&matchingThreshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numberOfPictures, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numberOfObjects, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Wait until all processes get the matching threshold, number of pictures, number of objects ans the objects
    MPI_Barrier(MPI_COMM_WORLD);

    // sending objects array to all processes
    if(rank == 0)     // master process                           
    {
        // send all objects to all processes
        for (int i = 1; i < size; i++)
            for (int j = 0; j < numberOfObjects; j++)
                sendObject(&objects[j], i, OBJECT_TAG);
    }
    else             // slave process
    {
        // allocating memory and recving all objects 
        objects = (Object *)malloc(numberOfObjects * sizeof(Object));
        checkErrorMemory(objects, "allocating memory for objects array in other processes");

        for (int i = 0; i < numberOfObjects; i++)
            receiveObject(&objects[i], 0, OBJECT_TAG, &status);
    }

    
    if (rank == 0)      // master process
    {   
        // send each process the first picture to work on
        for (int i = 1; i < size && picturesInProgress < numberOfPictures; i++)
        {
            MPI_Send(&picturesFinished, 1, MPI_INT, i, PICTURE_TAG, MPI_COMM_WORLD);
            sendPicture(&pictures[picturesInProgress], i, PICTURE_TAG);
            picturesInProgress++;
        }

        // while there are pictures to be processed
        while (picturesInProgress < numberOfPictures)
        {
            // receive processed picture
            receivePicture(&pictures[picturesFinished], MPI_ANY_SOURCE, PICTURE_TAG, &status);
            picturesFinished++;
            // send status more work to be done
            MPI_Send(&picturesFinished, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
            // send next picture to process
            sendPicture(&pictures[picturesInProgress], status.MPI_SOURCE, PICTURE_TAG);
            picturesInProgress++;
        }

        // receive the rest of the pictures
        while (picturesFinished < numberOfPictures)
        {
            receivePicture(&pictures[picturesFinished], MPI_ANY_SOURCE, PICTURE_TAG, &status);
            picturesFinished++;
        }

        // send terminate signal to all processes
        for (int i = 1; i < size; i++)
            MPI_Send(&picturesFinished, 1, MPI_INT, i, TERMINATE_TAG, MPI_COMM_WORLD);
        
        // saving results to file
        writeRes(OUTPUT_FILE, pictures, numberOfPictures);

        // free pictures array
        freePictures(pictures, numberOfPictures);
    }
    else                // other processes
    {
        MPI_Recv(&picturesFinished, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status) ;

        // while master process does not send terminate signal
        while (status.MPI_TAG != TERMINATE_TAG)
        {
            // allocating memory and recving picture 
            pictures = (Picture *)malloc(sizeof(Picture));
            checkErrorMemory(pictures, "allocating memory for picture in other processes");
            receivePicture(pictures, 0, MPI_ANY_TAG, &status);

            // search for objects
            findObjectsInPicture(pictures, objects, numberOfObjects, matchingThreshold);

            sendPicture(pictures, 0, PICTURE_TAG);
            freePictures(pictures, 1);
            
            MPI_Recv(&picturesFinished, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status) ;
        }
    }

    freeObjects(objects, numberOfObjects);
    
    if(rank == 0)
        printf("The whole program ran for %f sec \n", (MPI_Wtime() - timer));

    MPI_Finalize();
    return 0;
}