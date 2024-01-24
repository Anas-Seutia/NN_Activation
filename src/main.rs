use rand::Rng;
use std::io;
mod plotter;

const TRAINING_SET_SIZE: usize = 5;
const TESTING_SET_SIZE: usize = 5;
const LEARNING_RATE: f64 = 0.001;
const EPOCHS: usize = 5;

fn main() {
    loop {
        let mut rng = rand::thread_rng();


        // Initialize weights and biases
        let mut weights_ih: Vec<f64> = (0..4).map(|_| rng.gen_range(0.0..1.0)).collect(); // 2x2
        let mut weights_ho: Vec<f64> = (0..2).map(|_| rng.gen_range(0.0..1.0)).collect(); // 2x1
        let mut bias_h: Vec<f64> = (0..2).map(|_| rng.gen_range(0.0..1.0)).collect(); // 1x2
        let mut bias_o = rng.gen_range(0.0..1.0); // 1x1

        let epoch_list: [usize; EPOCHS] = core::array::from_fn(|i| i + 1);
        let mut errors:Vec<f32> = vec![0.0; EPOCHS];

        // Inputs and target
        let train_array: Vec<Vec<f64>> = (0..TRAINING_SET_SIZE).map(|_| vec![rng.gen_range(-2.5..2.5),rng.gen_range(-2.5..2.5)]).collect();
        let train_target: Vec<f64> = (0..TRAINING_SET_SIZE).map(|i| sigmoid(0.5 * train_array[i][0] + 1.5 * train_array[i][1])).collect();


        for epoch in 0..EPOCHS { // Training for EPOCHS epochs
            for i in 0..train_array.len() {
                let inputs = &train_array[i];
                let target = train_target[i];

                // Forward pass _________________________________________________________
                let hidden_input: Vec<f64> = mat_vec_mul(&weights_ih, &inputs, &bias_h);
                let hidden_output: Vec<f64> = hidden_input.iter().map(|&x| sigmoid(x)).collect();
                let final_input = dot_product(&weights_ho, &hidden_output) + bias_o;
                let final_output = final_input;

                // Compute the error
                let error = target - final_output;
                let output_prime = sigmoid_prime(final_output);
                let error_term_output = error * output_prime;

                errors[epoch] = error as f32;

                // Backward pass ________________________________________________________

                // Output to hidden
                let d_weights_ho: Vec<f64> = hidden_output.iter().map(|&x| LEARNING_RATE * error_term_output * x).collect();
                let d_bias_o = LEARNING_RATE * error_term_output;

                // Hidden to input
                let hidden_errors: Vec<f64> = weights_ho.iter().zip(hidden_output.iter()).map(|(&w, &o)| {
                    let hidden_prime = sigmoid_prime(o);
                    w * error_term_output * hidden_prime
                }).collect();

                // Hidden to input gradients
                let d_weights_ih: Vec<f64> = (0..4).map(|i| LEARNING_RATE * hidden_errors[i/2] * inputs[i%2]).collect();
                let d_bias_h: Vec<f64> = bias_h.iter().zip(hidden_errors.iter()).map(|(_, &e)| LEARNING_RATE * e).collect();

                // Update weights and biases
                update_vec(&mut weights_ho, &d_weights_ho);
                bias_o += d_bias_o;
                update_vec(&mut weights_ih, &d_weights_ih);
                update_vec(&mut bias_h, &d_bias_h);
        }
    }

        // Final output after training
        let test_array: Vec<Vec<f64>> = (0..TESTING_SET_SIZE).map(|_| vec![rng.gen_range(-2.5..2.5),rng.gen_range(-2.5..2.5)]).collect();
        let test_target: Vec<f64> = (0..TESTING_SET_SIZE).map(|i| sigmoid(0.5 * test_array[i][0] + 1.5 * test_array[i][1])).collect();

        let mut train_total_error: f64 = 0.0;
        let mut test_total_error: f64 = 0.0;

        println!("-----------------Training set----------------");
        for i in 0..train_array.len() {
            let inputs = &train_array[i];
            let target = train_target[i];
            let final_output = feed_forward(&weights_ih, &weights_ho, &bias_h, bias_o, &inputs);
            if i < test_target.len() {
                println!("Target: {:.5}, Output: {:.5}", target, final_output);
            }
            train_total_error += (target - final_output).powi(2);
        }

        println!("-----------------Testing set-----------------");
        for i in 0..test_array.len() {
            let inputs = &test_array[i];
            let target = test_target[i];
            let final_output = feed_forward(&weights_ih, &weights_ho, &bias_h, bias_o, &inputs);
            println!("Target: {:.5}, Output: {:.5}", target, final_output);
            test_total_error += (target - final_output).powi(2);
        }

        println!("-------------------Info set------------------");
        let train_mse = train_total_error / train_target.len() as f64;
        let test_mse = test_total_error / test_target.len() as f64;
        println!("Training set count: {}", TRAINING_SET_SIZE);
        println!("Testing set count: {}", TESTING_SET_SIZE);
        println!("Learning rate: {}", LEARNING_RATE);
        println!("Epochs: {}", EPOCHS);
        println!("Training MSE: {:.5}", train_mse);
        println!("Testing MSE: {:.5}", test_mse);

        println!("-------------Press Enter to Retry-------------");
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        print!("{}[2J", 27 as char);

        plotter::plot_epoch_accuracy(&epoch_list, &errors, "epoch_accuracy.png").unwrap();
    }
}

fn mat_vec_mul(matrix: &Vec<f64>, vector: &Vec<f64>, bias: &Vec<f64>) -> Vec<f64> {
    let mut result = vec![0.0; bias.len()];
    for i in 0..result.len() {
        result[i] = matrix[i*2] * vector[0] + matrix[i*2 + 1] * vector[1] + bias[i];
    }
    result
}

fn dot_product(vec1: &Vec<f64>, vec2: &Vec<f64>) -> f64 {
    vec1.iter().zip(vec2.iter()).map(|(&x, &y)| x * y).sum()
}

fn update_vec(vec: &mut Vec<f64>, delta: &Vec<f64>) {
    for (v, d) in vec.iter_mut().zip(delta.iter()) {
        *v += *d;
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_prime(y: f64) -> f64 {
    y * (1.0 - y)
}

fn feed_forward(weights_ih: &Vec<f64>, weights_ho: &Vec<f64>, bias_h: &Vec<f64>, bias_o: f64, inputs: &Vec<f64>) -> f64 {
    let hidden_input = mat_vec_mul(&weights_ih, &inputs, &bias_h);
    let hidden_output: Vec<f64> = hidden_input.iter().map(|&x| sigmoid(x)).collect();
    let final_input = dot_product(&weights_ho, &hidden_output) + bias_o;
    sigmoid(final_input) // Apply sigmoid to the output layer if you want to bound the final output between 0 and 1
}