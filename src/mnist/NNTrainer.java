package mnist;

import java.util.Arrays;
import java.util.List;

import neuralnetwork.NeuralNetwork;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.jet.math.Functions;

public class NNTrainer {
	static NeuralNetwork nn;
	public static void main(String[] args) {
		loadData();
		int nInputNodes = 28 * 28;
//		int nInputNodes = 2;
//		int nHiddenNodes = 2;
		int nOutputNodes = 10;
		double alfa = 0.1;
//		nn = new NeuralNetwork(nInputNodes, nHiddenNodes, nOutputNodes, alfa);
//		DoubleMatrix2D[] inputs = {new DenseDoubleMatrix2D(new double[][] {{1}, {1}}),
//								   new DenseDoubleMatrix2D(new double[][] {{0}, {0}}),
//								   new DenseDoubleMatrix2D(new double[][] {{0}, {1}}),
//								   new DenseDoubleMatrix2D(new double[][] {{1}, {0}})};
//		DoubleMatrix2D[] targets = {new DenseDoubleMatrix2D(new double[][] {{0}}),
//									new DenseDoubleMatrix2D(new double[][] {{0}}),
//									new DenseDoubleMatrix2D(new double[][] {{1}}),
//									new DenseDoubleMatrix2D(new double[][] {{1}})};
//		System.out.println(nn.hiddenWeights);
//		for(int i = 0; i < 10000; i++) {
//			for(int j = 0; j < 4; j++) {
//				nn.train(inputs[j], targets[j]);
//			}
//		}
//		System.out.println(nn.hiddenWeights);
//		System.out.println(nn.hiddenBiases);
//		System.out.println(nn.outputWeights);
//		System.out.println(nn.outputBias);
//		
//		for(int i = 0; i < 4; i++) {
//			System.out.println(nn.feedForward(inputs[i]));
//		}
		
		
		
//		nn = new NeuralNetwork(nInputNodes, new int[] {200}, nOutputNodes, alfa);
//		DoubleMatrix2D[] inputs = {new DenseDoubleMatrix2D(new double[][] {{1}, {1}}),
//								   new DenseDoubleMatrix2D(new double[][] {{0}, {0}}),
//								   new DenseDoubleMatrix2D(new double[][] {{0}, {1}}),
//								   new DenseDoubleMatrix2D(new double[][] {{1}, {0}})};
//		DoubleMatrix2D[] targets = {new DenseDoubleMatrix2D(new double[][] {{0}}),
//									new DenseDoubleMatrix2D(new double[][] {{0}}),
//									new DenseDoubleMatrix2D(new double[][] {{1}}),
//									new DenseDoubleMatrix2D(new double[][] {{1}})};
//		System.out.println(nn.weightsIH);
//		for(int i = 0; i < 10000; i++) {
//			for(int j = 0; j < 4; j++) {
//				nn.train2(inputs[j], targets[j]);
//			}
//		}
//		System.out.println(nn.weightsIH);
//		System.out.println(nn.biasH);
//		System.out.println(nn.weightsHO);
//		System.out.println(nn.biasO);
//		
//		
//		for(int i = 0; i < 4; i++) {
//			System.out.println(nn.feedForward2(inputs[i]));
//		}
		
		
//		test();
//		int epochs = 5;
//		double end = 0;
//		double start = System.currentTimeMillis();
//		for(int i = 1; i <= epochs; i++) {
//			train();	
//			end = System.currentTimeMillis();
//			System.out.println("Epoch: " + i + "/" + epochs +". Time: " + (end - start)/1000 + " seconds.");
//			start = System.currentTimeMillis();
//			test();
//		}
		
//		nn.saveData("NNweights_mnist_hidden250.txt");
		
		
//		int[][] forms = {{180, 180},
//						 {200, 200},
//						 {210, 210}, //[210, 210]: 89.0%, 10. Epoch, Best one!
//						 {220, 220},
//						 {180, 150},
//						 {200, 190},
//						 {210, 180},
//						 {220, 200}};
		int[][] forms = {{10, 10}};
		int[] f = {1, 2, 3};
		System.out.println(Arrays.toString(f));
//		testNeuralNetworks(forms);
		
		
	}
	
	private static void testNeuralNetworks(int[][] forms) {
		int nInputNodes = 28 * 28;
		int nOutputNodes = 10;
		double alfa = 0.1;
		NeuralNetwork nn;
		int epochs = 10;
		double maxPerformance;
		int bestEpoch;
		double performance;
		for(int[] hiddenLayers : forms) {
			nn = new NeuralNetwork(nInputNodes, hiddenLayers, nOutputNodes, alfa);
			maxPerformance = 0;
			bestEpoch = 0;
			for(int i = 1; i <= epochs; i++) {
				train(nn);
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
	private static void loadData() {
		String trainFileName = "mnist/mnist_train.csv";
		int nTrainImages = 1000; // Max 60000
		trainLoader.loadCSV(trainFileName, nTrainImages);
		trainLoader.initNormalizedInputs();
		trainLoader.initTargets();		
		
		String testFileName = "mnist/mnist_test.csv";
		int nTestImages = 500; // Max 10000
		testLoader.loadCSV(testFileName, nTestImages);
		testLoader.initNormalizedInputs();
		testLoader.initTargets();
		System.out.println("The data has been loaded.");
	
	}
	
	private static void train(NeuralNetwork nn) {
		List<DoubleMatrix2D> inputs = trainLoader.getNormalizedInputs();
		List<DoubleMatrix2D> targets = trainLoader.getTargets();
		for(int i = 1; i < inputs.size(); i++) {
			nn.train2(inputs.get(i), targets.get(i));
			if(i % 5000 == 0 && i != 0) {
				System.out.println("Images Trained: " +  i + "/" + inputs.size());
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
			int guess = maxIndex( nn.feedForward2(inputs.get(i)) );
			int target = maxIndex( targets.get(i) );
			if(guess == target) {
				correct ++;
			}
		}
		return (correct / targets.size() * 100.0);
	}
}








