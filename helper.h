#pragma once
#include <mpi.h>

#define INPUT_FILE "input.txt"
#define OUTPUT_FILE "output.txt"

#define PICTURE_TAG 0
#define OBJECT_TAG 1
#define TERMINATE_TAG 2

#define MSG_SIZE 30

// Structs

struct PictureStruct
{
    int ID;
    int dim;
    int *mat;
    int objFound;
    char* stats;  
};
typedef struct PictureStruct Picture;

struct ObjectStruct
{
    int ID;
    int dim;
    int *mat;
};
typedef struct ObjectStruct Object;




/*  checks if the allocation value is good prints an error message if it's not
    val = allocation value  |   msg = error message                                                             */ 
void checkErrorMemory(void* val, const char* msg);

/*  checks if the number of elements read is equal to the expected value, prints an error message if it's not
    val = read value    |   exp = expected value    |   msg = error message                                     */
void checkErrorRead(int val, int exp, const char* msg);

/* free memory for picture array
    pictures = array of pictures    |   numPictures = number of pictures in the array                           */
void freePictures(Picture* pictures, int numPictures);

/*  free memory for objects array
    objects = array of objects  |   numObjects = number of objects in the array                                 */
void freeObjects(Object* objects, int numObjects);

/*  reads the pixel matrix from the input file
    fp = file pointer   |   colorsMatrix = matrix pointer   |   dimension = matrix dimension                    */
void readingMatrix(FILE *fp, int *colorsMatrix, int dimension);

/*  reads all data from the input text file
    inputFile = files name  |   pictures = array of pictures    |   objects = array of objects  |   matchingThreshold = the matching threshold 
    numberOfPictures = the number of pictures   |   numberOfObjects = the number of objects                     */
void readInputFile(const char *inputFile, Picture **pictures, Object **objects, double *matchingThreshold, int *numberOfPictures, int *numberOfObjects);


/*  writes the results on the output file
    outputFile = name of the output file    |   pictures = array of pictures    |   numPictures = the number of pictures */  
void writeRes(const char *outputFile, Picture* pictures, int numPictures);

// ---------------------- MPI Functions -------------------------------

/*  send picture struct from a process to another process
    picture = picture struct to send    |   desRank = the receiving rank    |   tag = tag of action             */
void sendPicture(Picture *picture, int destRank, int tag);

/*  receive picture struct from another process to this process
    picture = picture struct to receive    |   sourceRank = the sending rank    |   tag = tag of action     |   status = status of the process      */
void receivePicture(Picture *picture, int sourceRank, int tag, MPI_Status *status);

/*  send object struct from a process to another process
    object = object struct to send    |   desRank = the receiving rank    |   tag = tag of action             */
void sendObject(Object *object, int destRank, int tag);

/*  receive picture struct from another process to this process
    picture = picture struct to receive    |   sourceRank = the sending rank    |   tag = tag of action     |   status = status of the process      */
void receiveObject(Object *object, int sourceRank ,int tag, MPI_Status *status);


// ---------------------- OpenMP Function -------------------------------

/*  calculates the matching between a picture and an object
    picture = picture to check  |   =objects = array of objects    |   numberOfObjects = the number of objects     |
    matchingThreshold = the matching threshold value                                                          */
void findObjectsInPicture(Picture *picture, Object *objects, int numberOfObjects, double matchingThreshold);

/*  add the results of matching indexs to picture stats
    picture = picture struct    |   object = object struct  |   res = array of the matching indexs   |   index = index of the match*/
void addResultsToPicture(Picture *picture, Object object, int* res, int index);


// ---------------------- CUDA Functions ---------------------------------

/*  calculates the matching between a picture and an object for all positions
    picture = picture struct    |   object = object struct  |   mmatchingThreshold = matching threshold value   |   foundFlag = flag if a match was found
    size = the amount of possible positions for a match                                                        */
extern void calculateMatchingOnGPU(Picture *picture, Object *object, double* matchingThreshold, int* foundFlag, int* indexsFound, int size);



