package mnist;

import java.util.Arrays;
import java.util.List;

import cern.colt.matrix.DoubleMatrix2D;
import neuralnetwork.NeuralNetwork;

public class NNTrainer {
	/*
	 *  Code used to create the weights and biases used in the application.
	 */
//	static NeuralNetwork nn;
//	public static void main(String[] args) {
//		loadData(60000, 10000);
//		int nInputNodes = 28 * 28;
//		int nOutputNodes = 10;
//		double alfa = 0.1;		
//		NeuralNetwork nn = new NeuralNetwork(nInputNodes, new Integer[]{240}, nOutputNodes, alfa); 
//		train(nn, 10);
//		System.out.println(test(nn));
//		nn.saveData("NNweights_mnist_240.txt");
//	}
	
	private static void testNeuralNetworks(Integer[][] forms) {
		int nInputNodes = 28 * 28;
		int nOutputNodes = 10;
		double alfa = 0.1;
		NeuralNetwork nn;
		int epochs = 2;
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
}


