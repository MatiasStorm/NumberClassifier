package gui;
import java.text.DecimalFormat;
import java.util.ArrayList;

import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import javafx.application.Application;
import javafx.event.EventHandler;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.SnapshotParameters;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Button;
import javafx.scene.image.Image;
import javafx.scene.image.PixelReader;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;
import javafx.scene.paint.Color;
import javafx.scene.text.Text;
import javafx.scene.transform.Transform;
import javafx.stage.Stage;
import mnist.MNIST;
import neuralnetwork.NeuralNetwork;

public class Main extends Application{
	private int width = 600;
	private int height = 300;
	private Canvas canvas;
	private GraphicsContext gc;
	BorderPane rootPane = new BorderPane();
	NeuralNetwork nn;
	public void start(Stage primaryStage) {
		String weightsFile = "NNweights_mnist_240.txt";
		
		nn = new NeuralNetwork(weightsFile, 0.1);
		
		Scene scene = new Scene(rootPane, width, height);
		scene.getStylesheets().add("gui/styleSheet.css");
		rootPane.setLeft(addCanvasPane());
		rootPane.setRight(addControlPane());
		
		primaryStage.setScene(scene);
		primaryStage.show();
		primaryStage.setTitle("Number Classifier");
		primaryStage.setResizable(false);
	}
	
	/**
	 * Initializes the canvas, adds it the a pane and returns the pane
	 * 
	 * @return canvasPane, instance of a pane containing a canvas
	 */
	private Pane addCanvasPane() {
		initCanvas();
		Pane canvasPane = new Pane();
		canvasPane.getChildren().add(canvas);
		
		return canvasPane;
	}
	
	/**
	 * Initializes the canvas, which is 10x larger than the MNIST images.
	 * 
	 * @return 
	 */
	private void initCanvas() {
		canvas = new Canvas(MNIST.imageWidth * 10, MNIST.imageHeight * 10);
		gc = canvas.getGraphicsContext2D();
		gc.setStroke(Color.BLACK);
		gc.setFill(Color.WHITE);
		gc.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
		gc.setLineWidth(28);
		canvas.addEventHandler(MouseEvent.MOUSE_PRESSED, new EventHandler<MouseEvent>() {
			@Override
			public void handle(MouseEvent event) {
				gc.beginPath();
				gc.moveTo(event.getX(), event.getY());
				gc.stroke();
			}
		});
		
		canvas.addEventHandler(MouseEvent.MOUSE_DRAGGED, new EventHandler<MouseEvent>() {
			@Override
			public void handle(MouseEvent event) {
				gc.lineTo(event.getX(), event.getY());
				gc.stroke();
				makeGuess();
			}
		});
	}
	
	/**
	 * Creates the control panel, and initializes the buttons with in it.
	 * 
	 * @return contorlPanel, instance of a panel containing a clear and a guess button. 
	 */
	private Pane addControlPane() {
		BorderPane controlPane = new BorderPane();
		HBox buttonPane = new HBox();
		buttonPane.setPadding(new Insets(10, 10, 10, 10));
		buttonPane.setSpacing(130);
		buttonPane.setAlignment(Pos.TOP_LEFT);
		
		Button clearButton = new Button("Clear");
		Button guessButton = new Button("Guess");
		buttonPane.getChildren().addAll(clearButton, guessButton);
		
		clearButton.setOnMouseClicked(e -> clearCanvas());
		guessButton.setOnMouseClicked(e -> makeGuess());
		
		controlPane.setBottom(buttonPane);
		controlPane.setCenter(addInfoPane());
		
		return controlPane;
	}
	
	
	
	/**
	 * Creats an info pane containing the text pane and the percentages pane
	 * 
	 * @return infoPane, instance of a border pane containing the text and percentage pane.
	 */
	private Pane addInfoPane() {
		BorderPane infoPane = new BorderPane();
		
		Pane textPane = getTextPane();
		Pane percentagePane = getPercentagePane();
		infoPane.setTop(textPane);
		infoPane.setBottom(percentagePane);
		
		return infoPane;
	}
	
	Text guessNumber;
	/**
	 * Setting up a text pane containing the guessed number
	 * 
	 * @return textPane, instance of a GridPane.
	 */
	private Pane getTextPane() {
		GridPane textPane = new GridPane();
		textPane.setHgap(15);
		textPane.setVgap(12);
		textPane.add(new Text("The Number Is: "), 0,0);
		guessNumber = new Text("  ");
		guessNumber.getStyleClass().add("guess-text");
		textPane.add(guessNumber, 1,0);
		return textPane;
	}
	
	ArrayList<Text> percentages = new ArrayList<>();
	/**
	 * Creates a GridPane containing the likely hood for the drawn number to be a number from 0 to 9, in percentages.
	 * 
	 * @return percentagePane, instance of a GridPane.
	 */
	private Pane getPercentagePane() {
		GridPane percentagePane = new GridPane();
		percentagePane.setHgap(30);
		percentagePane.add(new Text("Distribution"), 0, 0);
		for(int i = 1; i < 11; i++) {
			int index = i - 1;
			percentages.add(new Text("0.00%"));
			percentages.get(index).getStyleClass().add("percentage-text");
			if (i < 6) {
				percentagePane.add(new Text(i + ":"), 0, i);
				percentagePane.add(percentages.get(index), 1, i);
			} else {
				percentagePane.add(new Text(i + ":"), 2, i - 5);
				percentagePane.add(percentages.get(index), 3, i - 5);
			}
		}
		return percentagePane;
	}

	
	/**
	 * Clear all drawings on the canvas.
	 */
	private void clearCanvas() {
		gc.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
	}
	
	/**
	 * Calls the neural network to make a guess on what number the drawing represents and
	 * updates the gessNumber variable to display the guessed number in the infoPane.
	 */
	private void makeGuess() {
		DoubleMatrix2D input = convertCanvasToMatrix();
		DoubleMatrix2D guess = nn.feedForward(input);
		double max = -1;
		int maxIndex = -1;
		for(int i = 0; i < guess.rows(); i++) {
			DecimalFormat df = new DecimalFormat("##.##");
			percentages.get(i).setText( df.format(guess.get(i, 0)/guess.zSum() * 100) + "%");
			if(guess.get(i, 0) > max) {
				max = guess.get(i, 0);
				maxIndex = i;
			}
		}
		guessNumber.setText(Integer.toString(maxIndex));
	}
	
	/**
	 * Converts the canvas to a 28x28 matrix, which can be analyzed by the neural network.
	 * The matrix contains numbers between 0-1 which represents the pixel colors, white = 0, black = 1.
	 * 
	 * @return matrix, 28x28 matrix representation of the canvas where pixel colors are converted into a number between 0 and 1.
	 */
	private DoubleMatrix2D convertCanvasToMatrix() {
		SnapshotParameters sp = new SnapshotParameters();
		sp.setTransform(Transform.scale(0.1, 0.1));
		Image img = canvas.snapshot(sp, null);
		PixelReader pr = img.getPixelReader();
		
		DoubleMatrix2D matrix = new DenseDoubleMatrix2D(MNIST.nPixels, 1);
		int row = 0;
		for(int y = 0; y < img.getHeight(); y++) {
			for(int x = 0; x < img.getWidth(); x++) {
				double val = 1.0 - pr.getColor(x, y).getBrightness();
				matrix.set(row, 0, val);
				row++;
			}
		}		
		return matrix;
	}
}
