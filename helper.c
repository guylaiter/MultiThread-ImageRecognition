#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "helper.h"


void checkErrorMemory(void* val, const char* msg)
{
    if (val == NULL)
        {
            printf("ERROR: %s\n", msg);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
}

void checkErrorRead(int val, int exp, const char* msg)
{
    if (val != exp)
        {
            printf("ERROR: %s\n", msg);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
}

void readingMatrix(FILE *fp, int *colorsMatrix, int dimension)
{
    for (int j = 0; j < dimension; j++)
    {
        for (int k = 0; k < dimension; k++)
        {
            checkErrorRead(fscanf(fp, "%d", &colorsMatrix[j * dimension + k]), 1, "reading matrix from file");
        }
    }
}

void readInputFile(const char *inputFile, Picture **pictures, Object **objects, double *matchingThreshold, int *numberOfPictures, int *numberOfObjects)
{
    FILE *fp = fopen(inputFile, "r");

    checkErrorMemory(fp, "opening file for reading data");
    checkErrorRead(fscanf(fp, "%lf", matchingThreshold), 1, "reading matching thresh");
    
    // -------------- Reading pictures ----------------

    // read number of pictures
    checkErrorRead(fscanf(fp, "%d", numberOfPictures), 1, "reading number of pictures");


    // allocate memory for pictures array
    *pictures = (Picture *)malloc(*numberOfPictures * sizeof(Picture));
    checkErrorMemory(*pictures, "failed allocating memory 'picture array'");


    // read pictures from file and store them in pictures array of Picture structs
    for (int i = 0; i < *numberOfPictures; i++)
    {
        // read picture ID
        checkErrorRead(fscanf(fp, "%d", &(*pictures)[i].ID), 1, "reading picture ID");

        // read picture dimension
        checkErrorRead(fscanf(fp, "%d", &(*pictures)[i].dim), 1, "reading picture dim");

        // allocate memory for pixels matrix
        (*pictures)[i].mat = (int *)malloc((*pictures)[i].dim * (*pictures)[i].dim * sizeof(int));
        checkErrorMemory((*pictures)[i].mat, "allocating memory for picture matrix");

        readingMatrix(fp, (*pictures)[i].mat, (*pictures)[i].dim);

        (*pictures)[i].stats = (char*)malloc(sizeof(char) * MSG_SIZE);
        checkErrorMemory((*pictures)[i].stats, "allocating memory for 'picture stats'");
        sprintf((*pictures)[i].stats, "Picture %d: found Objects: ", (*pictures)[i].ID);

        (*pictures)[i].objFound = 0;
    }
    // -------------- Reading objects ----------------

    // read number of objects
    checkErrorRead(fscanf(fp, "%d", numberOfObjects), 1, "reading number of objects");

    // allocate memory for objects array
    *objects = (Object *)malloc(*numberOfObjects * sizeof(Object));
    checkErrorMemory( *objects, "allocating memory for 'objects array'");

    // read objects from file and store them in objects array of Object structs
    for (int i = 0; i < *numberOfObjects; i++)
    {
        // read object ID
        checkErrorRead(fscanf(fp, "%d", &(*objects)[i].ID), 1, "reading object ID");

        // read object dimension
        checkErrorRead(fscanf(fp, "%d", &(*objects)[i].dim), 1, "reading object dimension");

        // allocate memory for colors matrix
        (*objects)[i].mat = (int *)malloc((*objects)[i].dim * (*objects)[i].dim * sizeof(int));
        checkErrorMemory((*objects)[i].mat, "allocating memory for picture matrix");

        readingMatrix(fp, (*objects)[i].mat, (*objects)[i].dim);
    }
}

void sendPicture(Picture *picture, int destRank, int tag)
{
    MPI_Send(&picture->ID, 1, MPI_INT, destRank, tag, MPI_COMM_WORLD);                          // ID
    MPI_Send(&picture->dim, 1, MPI_INT, destRank, tag, MPI_COMM_WORLD);                         // dim
    MPI_Send(picture->mat, picture->dim * picture->dim, MPI_INT, destRank, tag, MPI_COMM_WORLD);// mat
    MPI_Send(&picture->objFound, 1, MPI_INT, destRank, tag, MPI_COMM_WORLD);                    // objects found

    int sizeOfStats = strlen(picture->stats) + 1;
    MPI_Send(&sizeOfStats, 1, MPI_INT, destRank, tag, MPI_COMM_WORLD);                          // size of stats string
    MPI_Send(picture->stats, sizeOfStats, MPI_CHAR, destRank, tag, MPI_COMM_WORLD);             // stats
}

void receivePicture(Picture *picture, int sourceRank, int tag, MPI_Status *status)
{
    MPI_Recv(&picture->ID, 1, MPI_INT, sourceRank, tag, MPI_COMM_WORLD, status);                            // ID
    MPI_Recv(&picture->dim, 1, MPI_INT, sourceRank, tag, MPI_COMM_WORLD, status);                           // dim

    // allocate memory for picture matrix
    picture->mat = (int *)malloc(picture->dim * picture->dim * sizeof(int));
    checkErrorMemory(picture->mat, "allocating memory for picture matrix receive");

    MPI_Recv(picture->mat, picture->dim * picture->dim, MPI_INT, sourceRank, tag, MPI_COMM_WORLD, status);  // mat
    MPI_Recv(&picture->objFound, 1, MPI_INT, sourceRank, tag, MPI_COMM_WORLD, status);                      // objects found

    int sizeOfStats;
    MPI_Recv(&sizeOfStats, 1, MPI_INT, sourceRank, tag, MPI_COMM_WORLD, status);                            // size of stats string

    // allocate memory for picture stats
    picture->stats = (char*)malloc(sizeOfStats * sizeof(char));
    checkErrorMemory(picture->stats, "allocating memory for picture stats receive");

    MPI_Recv(picture->stats, sizeOfStats, MPI_CHAR, sourceRank, tag, MPI_COMM_WORLD, status);               // stats

}

void sendObject(Object *object, int destRank, int tag)
{
    MPI_Send(&object->ID, 1, MPI_INT, destRank, tag, MPI_COMM_WORLD);
    MPI_Send(&object->dim, 1, MPI_INT, destRank, tag, MPI_COMM_WORLD);
    MPI_Send(object->mat, object->dim * object->dim, MPI_INT, destRank, tag, MPI_COMM_WORLD);
}

void receiveObject(Object *object, int sourceRank ,int tag, MPI_Status *status)
{
    MPI_Recv(&object->ID, 1, MPI_INT, sourceRank, tag, MPI_COMM_WORLD, status);
    MPI_Recv(&object->dim, 1, MPI_INT, sourceRank, tag, MPI_COMM_WORLD, status);

    // allocate memory for object matrix
    object->mat = (int *)malloc(object->dim * object->dim * sizeof(int));
    checkErrorMemory(object->mat, "allocating memory for object matrix receive");
    MPI_Recv(object->mat, object->dim * object->dim, MPI_INT, sourceRank, tag, MPI_COMM_WORLD, status);
}

void writeRes(const char *outputFile, Picture* pictures, int numPictures)
{
    // Open file
    FILE *fp = fopen(outputFile, "w");
    checkErrorMemory(fp , "opening file for writing results");

    // write results to file
    for (int i = 0 ; i < numPictures; i++)
    {
        if(pictures[i].objFound > 2)
        {
            fprintf(fp, "%s\r\n", pictures[i].stats);
        }
        else
        {
            fprintf(fp, "Picture %d: No three different Objects were found\r\n", pictures[i].ID);
        }
    }
    fclose(fp);
}

void findObjectsInPicture(Picture *picture, Object *objects, int numberOfObjects, double matchingThreshold)
{
    #pragma omp parallel for
    for (int i = 0; i < numberOfObjects; i++)
    {
        int size = (picture->dim - objects[i].dim + 1) * (picture->dim - objects[i].dim + 1);
        int foundFlag = 0;
        int* indexsFound = (int*)calloc(2 * size, sizeof(int));
        *(indexsFound) = 1;

        #pragma omp task 
        {
            calculateMatchingOnGPU(picture, &objects[i], &matchingThreshold, &foundFlag, indexsFound, size);
        }

        #pragma omp taskwait

        for (int j = 0; j < 2 * size; j += 2)
        {
            if (j == 0 && *(indexsFound) == 0)
            {
                #pragma omp task 
                {
                    addResultsToPicture(picture, objects[i], indexsFound, j);
                }
            }
            else if (j != 0 && (*(indexsFound + j) != 0 || *(indexsFound + j + 1) != 0))
            {
                #pragma omp task 
                {
                    addResultsToPicture(picture, objects[i], indexsFound, j);
                }
            }
        }

        #pragma omp taskwait
        free(indexsFound);
    }
}

void addResultsToPicture(Picture *picture, Object object, int* res, int index)
{
    char temp[MSG_SIZE];
    sprintf(temp, "%d Position (%d,%d); ", object.ID, *(res + index), *(res + index + 1));

    (*picture).stats = (char*)realloc((*picture).stats, (int)strlen((*picture).stats) + MSG_SIZE);

    strcat((*picture).stats, temp);        
}

void freePictures(Picture* pictures, int numPictures)
{
    for (int i = 0; i < numPictures; i ++)
    {
        free(pictures[i].stats);
        free(pictures[i].mat);
    }
    free(pictures);
}

void freeObjects(Object* objects, int numObjects)
{
    for (int i = 0; i < numObjects; i ++)
        free(objects[i].mat);

    free(objects);
}
