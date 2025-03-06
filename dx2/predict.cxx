#include <dx2/experiment.h>
// #include <fmt/color.h>
// #include <fmt/core.h>
// #include <fmt/os.h>
// #include <hdf5.h>

#include <argparse/argparse.hpp>
#include <chrono>
// #include <dx2/h5/h5read_processed.hpp>
// essential; common.hpp is in ffs => predict.cxx cannot be build on dx2 alone
#include <common.hpp>
#include <cstdlib>
#include <dx2/h5/h5write.hpp>
#include <exception>
#include <fstream>
#include <iostream>  // Debugging
#include <thread>
#include <vector>  // Unused so far...

/**
 * @brief Takes a default-initialized ArgumentParser object and configures it 
 *      with the arguments to be parsed; also assigns various properties to each 
 *      argument, eg. help message, default value, etc.
 * 
 * @param parser The ArgumentParser object (pre-input) to be configured.
 */
void configure_parser(argparse::ArgumentParser& parser) {
    parser.add_argument("-e", "--expt").help("Path to DIALS expt file").required();
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
void verify_arguments(const argparse::ArgumentParser& parser) {
    if (!parser.is_used("expt")) {
        logger->error("Must specify experiment list file with --expt\n");
        std::exit(1);
    }
    // FIXME use highest resolution by default to remove this requirement.
    if (!parser.is_used("dmin")) {
        logger->error("Must specify --dmin\n");
        std::exit(1);
    }
    if (parser.is_used("nthreads") && parser.get<size_t>("nthreads") < 1) {
        logger->error("--nthreads cannot be less than 1\n");
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    auto t1 = std::chrono::system_clock::now();
    auto parser = argparse::ArgumentParser();
    configure_parser(parser);

    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& err) {
        logger->error(err.what());
        std::exit(1);
    }

    verify_arguments(parser);

    // Obtain argument values from the command line
    auto expt_path = parser.get<std::string>("expt");
    auto dmin = parser.get<float>("dmin");
    auto static_predict = parser.get<bool>("static_predict");
    auto dynamic_shadows = parser.get<bool>("dynamic_shadows");
    auto buffer_size = parser.get<size_t>("buffer_size");
    auto nthreads = parser.get<size_t>("nthreads");

    // FIXME: What do the two macros below mean?
    // hid_t file = H5Fopen(expt_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    // std::cout << file << '\n';

    // Flatten experiments into a 1D array
    /*
    experiments = flatten_experiments(params.input.experiments)
    */

    // If length of experiments in 0, print a help message and exit
    /*
    if len(experiments) == 0:
        parser.print_help()
        return
    */

    // Create an empty reflection table to store predictions in
    /*
    predicted_all = reflection_table()
    */

    // Populate `predicated_all` with predictions
    /*
    for i_expt, expt in enumerate(experiments):
    */

    // Predict reflections outside the range of the scan if buffer_size > 0
    /*
        if params.buffer_size > 0:
            expt.scan.set_image_range(
                (
                    expt.scan.get_image_range()[0] - params.buffer_size,
                    expt.scan.get_image_range()[1] + params.buffer_size,
                )
            )
            expt.scan.set_oscillation(
                (
                    expt.scan.get_oscillation()[0] - params.buffer_size * oscillation[1],
                    expt.scan.get_oscillation()[1],
                )
            )
        */

    // Create reflection table using `expt`, `params.force_static`, `params.d_min`
    /*
        predicted = flex.reflection_table.from_predictions(
            expt, force_static=params.force_static, dmin=params.d_min
        )
        */

    // Copy experiment identifiers verbatim
    /*
        predicted.experiment_identifiers()[i_expt] = experiments[i_expt].identifier
        predicted["id"] = flex.int(len(predicted), i_expt)
        predicted_all.extend(predicted)
        */

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

    // Try to find bounding boxed for each experiment
    /*
    try:
        predicted_all.compute_bbox(experiments)
    except Exception:
        pass
    */

    // Save reflections to file
    /*
    Command.start(f"Saving {len(predicted_all)} reflections to {params.output}")
    predicted_all.as_file(params.output)
    Command.end(f"Saved {len(predicted_all)} reflections to {params.output}")
    */

    auto t2 = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_time = t2 - t1;
    logger->info("Total time for prediction: {:.4f}s", elapsed_time.count());
    return 0;
}