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
	
	private Pane addCanvasPane() {
		initCanvas();
		Pane canvasPane = new Pane();
		canvasPane.getChildren().add(canvas);
		
		return canvasPane;
	}
	
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
	
	Text guessText;
	ArrayList<Text> percentagesText = new ArrayList<>();
	private Pane addInfoPane() {
		BorderPane infoPane = new BorderPane();
		GridPane textPane = new GridPane();
		textPane.setHgap(15);
		textPane.setVgap(12);
		textPane.add(new Text("The Number Is: "), 0,0);
		guessText = new Text("  ");
		guessText.getStyleClass().add("guess-text");
		textPane.add(guessText, 1,0);
		textPane.add(new Text("Distribution:"), 0, 1, 2, 1);
		
		GridPane percentagePane = new GridPane();
		percentagePane.setHgap(30);
		for(int i = 0; i < 10; i++) {
			percentagesText.add(new Text("0.00%"));
			percentagesText.get(i).getStyleClass().add("percentage-text");
			if (i < 5) {
				percentagePane.add(new Text(i + ":"), 0, i);
				percentagePane.add(percentagesText.get(i), 1, i);
			} else {
				percentagePane.add(new Text(i + ":"), 2, i - 5);
				percentagePane.add(percentagesText.get(i), 3, i - 5);
			}
		}
		infoPane.setTop(textPane);
		infoPane.setBottom(percentagePane);
		
		return infoPane;
	}
	
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
	
	private void clearCanvas() {
		gc.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
	}
	
	private void makeGuess() {
		DoubleMatrix2D input = convertCanvasToMatrix();
		DoubleMatrix2D guess = nn.feedForward(input);
		double max = -1;
		int maxIndex = -1;
		for(int i = 0; i < guess.rows(); i++) {
			DecimalFormat df = new DecimalFormat("##.##");
			percentagesText.get(i).setText( df.format(guess.get(i, 0)/guess.zSum() * 100) + "%");
			if(guess.get(i, 0) > max) {
				max = guess.get(i, 0);
				maxIndex = i;
			}
		}
		guessText.setText(Integer.toString(maxIndex));
	}
	
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
