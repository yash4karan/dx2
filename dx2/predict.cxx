#include <dx2/experiment.h>
#include <dx2/reflection.h>
// #include <fmt/color.h>
// #include <fmt/core.h>
// #include <fmt/os.h>
// #include <hdf5.h>

#include <argparse/argparse.hpp>
#include <chrono>
// essential; common.hpp is in ffs => predict.cxx cannot be build on dx2 alone
#include <common.hpp>
#include <cstdlib>
// #include <dx2/h5/h5read_processed.hpp>
// #include <dx2/h5/h5write.hpp>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>  // Debugging
#include <thread>
#include <vector>  // Unused so far...

// FIXME This is just a placeholder class to give an intution on how reflection tables work.
// It is desigened only to support the API used by dx2.predict
class ReflectionTable {
  public:
    ReflectionTable() = default;
    ReflectionTable(std::string& h5_filepath) {}
    void add_table(ReflectionTable refl);
};

template <typename T>
ReflectionTable predict_from_expt(Experiment<T> expt, bool static_predict, float dmin) {
    return ReflectionTable();
}

/**
 * @brief Takes a default-initialized ArgumentParser object and configures it 
 *      with the arguments to be parsed; also assigns various properties to each 
 *      argument, eg. help message, default value, etc.
 * 
 * @param parser The ArgumentParser object (pre-input) to be configured.
 */
void configure_parser(argparse::ArgumentParser& parser) {
    parser.add_argument("-e", "--expt")
      .help("Path to input experiment file")
      .required();
    parser.add_argument("-o", "--output")
      .help("Path to output reflection file")
      .default_value("predicted.refl")
      .implicit_value("predicted.refl");
    parser.add_argument("--dmin")
      .help("Minimum d-spacing of predicted reflections")
      .scan<'f', float>()
      .required();
    parser.add_argument("-s", "--static_predict")
      .help("For a scan varying model, force static prediction")
      .default_value(false)
      .implicit_value(true);
    // The below is the opposite of ignore_shadows used in DIALS
    // This configuration allows for natural implicit-value flagging.
    parser.add_argument("-d", "--dynamic_shadows")
      .help("Enable dynamic shadowing")
      .default_value(false)
      .implicit_value(true);
    parser.add_argument("-b", "--buffer_size")
      .help(
        "Calculate predictions within a buffer zone of n images either side"
        "of the scan")
      .scan<'u', size_t>()
      .default_value<size_t>(0);
    parser.add_argument("-n", "--nthreads")
      .help(
        "The number of threads to use for the fft calculation."
        "Defaults to the value of std::thread::hardware_concurrency."
        "Better performance can typically be obtained with a higher number"
        "of threads than this.")
      .scan<'u', size_t>()
      .default_value<size_t>(std::thread::hardware_concurrency());
}

/**
 * @brief Takes an ArgumentParser object after the user has entered input and checks 
 *      it for consistency; outputs errors and exits the program if a check fails.
 * 
 * @param parser The ArgumentParser object (post-input) to be verified.
 */
void verify_parser_arguments(const argparse::ArgumentParser& parser) {
    if (!parser.is_used("expt")) {
        logger->error("Must specify experiment list file with --expt\n");
        std::exit(1);
    }
    // FIXME use highest resolution by default to remove this requirement.
    if (!parser.is_used("dmin")) {
        logger->error("Must specify --dmin\n");
        std::exit(1);
    }
    if (parser.get<int>("buffer_size") < 0) {
        logger->error("--buffer_size cannot be less than 0\n");
        std::exit(1);
    }
    if (parser.get<size_t>("nthreads") < 1) {
        logger->error("--nthreads cannot be less than 1\n");
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    auto t1 = std::chrono::system_clock::now();
    // Declare and configure the parser, then parse the arguments and verify them for consistency.
    auto parser = argparse::ArgumentParser();
    configure_parser(parser);
    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& err) {
        logger->error(err.what());
        std::exit(1);
    }
    verify_parser_arguments(parser);

    // Obtain argument values from the command line
    auto expt_path = parser.get<std::string>("expt");
    auto output_path = parser.get<std::string>("output");
    auto dmin = parser.get<float>("dmin");
    auto static_predict = parser.get<bool>("static_predict");
    auto dynamic_shadows = parser.get<bool>("dynamic_shadows");
    auto buffer_size = parser.get<int>("buffer_size");
    auto nthreads = parser.get<size_t>("nthreads");

    // FIXME: What do the two macros below mean?
    // h5read_handle file = H5Fopen(expt_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    // std::cout << file << '\n';

    // FIXME: Flatten experiments into a 1D array
    /*
    experiments = flatten_experiments(params.input.experiments)
    */
    // FIXME: At this stage, assume that `experiments` has type std::vector<Experiment>
    auto experiments = std::vector<Experiment<float>>(100);

    // If length of experiments in 0, print a help message and exit
    if (!experiments.size()) {
        logger->error("Experiment list is empty.\n");
        std::exit(1);
    }

    // Create an empty reflection table to store predictions in
    /*
    predicted_all = reflection_table()
    */
    // FIXME: This is a placeholder for the actual reflection table.
    ReflectionTable predicted_all;

    // Populate `predicated_all` with predictions
    for (int i = 0; i < experiments.size(); ++i) {
        Experiment expt{experiments[i]};
        // Predict reflections outside the range of the scan if buffer_size > 0
        if (buffer_size > 0) {
            // FIXME: This code assumes that `scan()` returns a reference to the _scan object in expt; this is NOT the case at the time of writing.
            std::array<int, 2> image_range{expt.scan().get_image_range()};
            std::array<double, 2> oscillation{expt.scan().get_oscillation()};
            expt.scan() =
              Scan({image_range[0] - buffer_size, image_range[1] + buffer_size},
                   {oscillation[0] - buffer_size * oscillation[1], oscillation[1]});
        }

        // Write prediction algorithm here...
        // Create reflection table using `expt`, `params.force_static`, `params.d_min`
        /*
        predicted = flex.reflection_table.from_predictions(
            expt, force_static=params.force_static, dmin=params.d_min
        )
        */
        ReflectionTable predicted = predict_from_expt(expt, static_predict, dmin);

        // FIXME: This code assumes resonable API for `predicted` and `predicted_all`. These classes do not exist yet.
        // FIXME: This code assumes that the Experiment class contains a method `identifier()`. These classes do not exist yet.
        predicted.experiment_identifiers()[i] = experiments[i].identifier();
        // FIXME: The "id" might have different API; the data type of the "id" might be different.
        predicted["id"] = std::array<int, 2>{len(predicted), i};
        predicted_all.add_table(predicted);
    }

    // If not ignoring shadows, look for reflections in the masked region
    /*
    if not params.ignore_shadows:
        try:
            experiments = ExperimentListFactory.from_json(
                experiments.as_json(), check_format=True
            )
        except OSError as e:
            sys.exit(
                f"Unable to read image data. Please check {e.filename} is accessible"
            )
        shadowed = filter_shadowed_reflections(
            experiments, predicted_all, experiment_goniometer=True
        )
        predicted_all = predicted_all.select(~shadowed)
    */
    if (dynamic_shadows) {
        // FIXME: Assuming that `experiments` is a Experiment-dictionary that is already conformant with JSON
    }

    // Try to find bounding boxed for each experiment
    // FIXME: `ReflectionTable` does not have a member function `compute_bbox()` as of now.
    try {
        predicted_all.compute_bbox(experiments);
    } catch (std::exception& err) {
        logger->info(err.what());
        continue;
    }

    // Save reflections to file
    logger->info("Saving {} reflections to {}", predicted_all.size(), output_path);
    // FIXME: The following call needs to be modified as the dataset path is included within the Reflection Table. An overload will do.
    write_data_to_h5_file(output_path, dataset_path, predicted_all);
    logger->info("Saved {} reflections to {}", predicted_all.size(), output_path);

    auto t2 = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_time = t2 - t1;
    logger->info("Total time for prediction: {:.4f}s", elapsed_time.count());
    return 0;
}