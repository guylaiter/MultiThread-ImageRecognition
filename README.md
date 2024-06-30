<!DOCTYPE html>
<html>
<head>
</head>
<body>
	<h1>MultiThread-ImageRecognition</h1>
	<p> This is a parallelized implementation of a simplified image recognition algorithm using MPI, OpenMP, and CUDA. 🔍🖼️</p>
	<h2>Project Description</h2>
	<p>📝 The project deals with sets of pictures and objects of different sizes. Each matrix member represents a color, and the range of possible colors is [1, 100]. The project's goal is to find an object within a picture using a matching algorithm that calculates the total difference between overlapping members of the picture and the object.</p>
	<h2>Technologies Used</h2>
	<ul>
		<b>MPI (Message Passing Interface)</b>
    <p>Master-Slave Model: The master process (rank 0) dynamically allocates pictures to the available slave processes. When a slave process is free, the master sends a new picture for processing from the picture pool.
    Process Updates: Each slave process updates the statistics for its assigned picture and returns the results to the master.</p>
		<b>OpenMP - Open Multi-Processing</b>
    <p>used to divide the objects on each picture into OpenMP threads. Each thread checks a specific object on the picture.</p>
		<b>CUDA</b>
    <p>Compute Unified Device Architecture, used to calculate all possible locations of the given object on the picture in parallel.</p>
	</ul>
	<h2>Requirements</h2>
	<p>📋 To run this project, you will need:</p>
	<ul>
		<li>C++ compiler with OpenMP support 🖥️💻</li>
		<li>MPI library 📚</li>
		<li>CUDA Toolkit 🛠️</li>
	</ul>
	<h2>Output Format</h2>
	<p>📄 The output file will contain the results of the recognition algorithm for each picture. For each picture, the log will indicate whether at least three objects were found with an appropriate matching value. If three objects were found, the log will also include the starting position of each object in the picture.</p>
  <h2>Performance</h2>
<p>The parallelized implementation using MPI, OpenMP, and CUDA significantly improves the performance of the image recognition algorithm. The original sequential implementation took approximately 60 seconds to complete, while the parallelized implementation reduced this time to approximately 0.52 seconds.</p>
</body>
</html>
