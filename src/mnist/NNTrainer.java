package mnist;

import java.util.Arrays;
import java.util.List;

import cern.colt.matrix.DoubleMatrix2D;
import neuralnetwork.NeuralNetwork;
import mnist.MNIST;

public class NNTrainer {
	static NeuralNetwork nn;
	static int nInputNodes = MNIST.nPixels;
	static int nOutputNodes = MNIST.nLabels;
	static double alfa = 0.1;	
	static int epochs = 1;
	/**
	 * Main method used to initialize neural networks or train multiple neural networks at once.
	 * @param args
	 */
	public static void main(String[] args) {
		loadData(60000, 10000);
		
		Integer[] hiddenLayers = {240}; // Modify this to change the hidden layers of the network.
		
		NeuralNetwork nn = new NeuralNetwork(nInputNodes, hiddenLayers, nOutputNodes, alfa); 
		train(nn, epochs);
		
		System.out.println(test(nn));
//		nn.saveData("NNweights_mnist_240.txt");   // Saves the neural network to the supplied file name.
	}
	
	/**
	 * Tests multiple neural networks at once, while printing out performance information about each network.
	 * Can only test neural networks for the MNIST data set.
	 * 
	 * @param  forms - a 2d array containing the hidden layes of each of the neural networks, example:
	 * 					forms =[ [20, 20],      - Neural network with two hidden layers with 20 nodes in each.
	 * 							 [10, 20, 10] ] - Network with three hidden layers, two with 10 nodes and one with 20.
	 * @return
	 */
	private static void testNeuralNetworks(Integer[][] forms) {
		NeuralNetwork nn;
		double maxPerformance;
		int bestEpoch;
		double performance;
		for(Integer[] hiddenLayers : forms) {
			nn = new NeuralNetwork(nInputNodes, hiddenLayers, nOutputNodes, alfa);
			maxPerformance = 0;
			bestEpoch = 0;
			for(int i = 1; i <= epochs; i++) {
				train(nn, 1);
				if ( (performance = test(nn)) > maxPerformance){
					maxPerformance = performance;
					bestEpoch = i;
				}
			}
			System.out.println(Arrays.toString(hiddenLayers) + ": " + maxPerformance + "%, " + bestEpoch + ". Epoch");
		}
	}
	
	static MNISTLoader trainLoader = new MNISTLoader();
	static MNISTLoader testLoader = new MNISTLoader();
	/**
	 * Loads the MNIST data from the "mnist/" folder, into two instances of the MNISTLoader class,
	 * called trainLoader and testLoader.
	 * 
	 * @param  nTrainingImages	- Number of images to train the neural network on.
	 * @param  nTestImages		- number of images to test the neural network on.
	 * @return
	 */
	private static void loadData(int nTrainingImages, int nTestImages) {
		String trainFileName = "mnist/mnist_train.csv";
		int nTrainImages = Math.min(nTrainingImages, 60000); // Max 60000
		trainLoader.loadCSV(trainFileName, nTrainImages);
		trainLoader.initNormalizedInputs();
		trainLoader.initTargets();		
		
		String testFileName = "mnist/mnist_test.csv";
		nTestImages = Math.min(nTestImages, 10000); // Max 10000
		testLoader.loadCSV(testFileName, nTestImages);
		testLoader.initNormalizedInputs();
		testLoader.initTargets();
		System.out.println("The data has been loaded.");
	
	}
	
	
	/**
	 * Trains the neural network, for the supplied number of epochs.
	 * It prints a message for each 5000th images trained on.
	 * 
	 * @param nn 	 - The neural network one wants to train.
	 * @param epochs - The number of epochs the given network should be trained for.
	 * @return
	 */
	private static void train(NeuralNetwork nn, int epochs) {
		List<DoubleMatrix2D> inputs = trainLoader.getNormalizedInputs();
		List<DoubleMatrix2D> targets = trainLoader.getTargets();
		for(int epoch = 1; epoch <= epochs; epoch++) {
			for(int i = 1; i < inputs.size(); i++) {
				nn.train(inputs.get(i), targets.get(i));
				if(i % 5000 == 0 && i != 0) {
					System.out.println("Images Trained: " +  i + "/" + inputs.size() + " - Epoch: " + epoch);
				}
			}
		}
	}
	
	
	/**
	 * Tests the neural network against the test data supplied by the testLoader.
	 * 
	 * @param  nn - The neural network one wants to test.
	 * @return The percentage of correctly classified images.
	 */
	private static double test(NeuralNetwork nn) {
		List<DoubleMatrix2D> inputs = testLoader.getNormalizedInputs();
		List<DoubleMatrix2D> targets = testLoader.getTargets();
		double correct = 0;
		for(int i = 0; i < inputs.size(); i++) {
			int guess = maxIndex( nn.feedForward(inputs.get(i)) );
			int target = maxIndex( targets.get(i) );
			if(guess == target) {
				correct ++;
			}
		}
		return (correct / targets.size() * 100.0);
	}
	
	
	/**
	 * Finds and returns the row index of the highest number in a DoubleMatrix2D containing a single column.
	 * 
	 * @param  m 	 - A DoubleMatrix2D containing a single column.
	 * @return index - The row index of the highest number.
	 */
	private static int maxIndex(DoubleMatrix2D m) {
		int index = -1;
		double max = -999;
		for(int i = 0; i < m.rows(); i++) {
			if (m.get(i, 0) > max) {
				index = i;
				max = m.get(i, 0);
			}
		}
		return index;
	}
}


