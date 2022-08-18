#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <Exceptions.h>
#include <ImageIO.h>
// #include <ImagesCPU.h>
// #include <ImagesNPP.h>

#include <string>
// #include <fstream>
#include <iostream>
#include <filesystem>

#include <cuda_runtime.h>
// #include <npp.h>

#include <helper_cuda.h>
// #include <helper_string.h>

bool printfNPPinfo(int argc, char *argv[])
{
    const NppLibraryVersion *libVer = nppGetLibVersion();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor,
           libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
           (driverVersion % 100) / 10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
           (runtimeVersion % 100) / 10);

    // Min spec is SM 1.0 devices
    bool bVal = checkCudaCapabilities(1, 0);
    return bVal;
}

void cannyFilter(const std::string &filePath, const std::string &outputFile)
{
    try
    {
        std::cout << "Processing of " << filePath << " started." << std::endl;
        npp::ImageCPU_8u_C1 hostSrc;
        npp::loadImage(filePath, hostSrc);
        npp::ImageNPP_8u_C1 deviceSrc(hostSrc);
        const NppiSize srcSize = {(int)deviceSrc.width(), (int)deviceSrc.height()};
        const NppiPoint srcOffset = {0, 0};

        const NppiSize filterROI = {(int)deviceSrc.width(), (int)deviceSrc.height()};
        npp::ImageNPP_8u_C1 deviceDst(filterROI.width, filterROI.height);

        const Npp16s lowThreshold = 72;
        const Npp16s highThreshold = 256;

        int bufferSize = 0;
        Npp8u *scratchBufferNPP = 0;
        NPP_CHECK_NPP(nppiFilterCannyBorderGetBufferSize(filterROI, &bufferSize));
        cudaMalloc((void **)&scratchBufferNPP, bufferSize);

        if ((bufferSize > 0) && (scratchBufferNPP != 0))
        {
            NPP_CHECK_NPP(nppiFilterCannyBorder_8u_C1R(deviceSrc.data(), deviceSrc.pitch(), srcSize, srcOffset,
                                                       deviceDst.data(), deviceDst.pitch(), filterROI,
                                                       NppiDifferentialKernel::NPP_FILTER_SOBEL, NppiMaskSize::NPP_MASK_SIZE_3_X_3,
                                                       lowThreshold, highThreshold, nppiNormL2, NPP_BORDER_REPLICATE,
                                                       scratchBufferNPP));
        }

        cudaFree(scratchBufferNPP);

        npp::ImageCPU_8u_C1 hostDst(deviceDst.size());
        deviceDst.copyTo(hostDst.data(), hostDst.pitch());
        saveImage(outputFile, hostDst);
        std::cout << "Processing of " << filePath << " ended. Result saved to: " << outputFile << std::endl;

        nppiFree(deviceSrc.data());
        nppiFree(deviceDst.data());
    }
    catch (npp::Exception &rException)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rException << std::endl;
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
    }
    catch (...)
    {
        std::cerr << "Program error! An unknow type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
    }
}

void applyFilter(const std::string &filterType, const std::string &filePath, const std::string &outputFile)
{
    if (filterType == "canny")
    {
        cannyFilter(filePath, outputFile);
    }
    else
    {
        std::cout << "Filter type isn't supported!" << std::endl;
    }
}

int main(int argc, char *argv[])
{
    printf("%s Starting...\n\n", argv[0]);

    if (!printfNPPinfo(argc, argv))
    {
        exit(EXIT_SUCCESS);
    }

    findCudaDevice(argc, (const char **)argv);

    char *inputData;
    if (argc >= 2)
    {
        inputData = argv[1];
        if (!inputData)
        {
            std::cerr << "Cannot read the input data!" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        std::cerr << "Input folder or image missed!" << std::endl;
        exit(EXIT_FAILURE);
    }

    char *filterData;
    if (argc >= 3)
    {
        filterData = argv[2];
        if (!inputData)
        {
            std::cerr << "Cannot read the filter type!" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        std::cerr << "Filter type is missed!" << std::endl;
        exit(EXIT_FAILURE);
    }

    char *outputData;
    if (argc >= 4)
    {
        outputData = argv[3];
        if (!inputData)
        {
            std::cerr << "Cannot read the filter type!" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        std::cerr << "Filter type is missed!" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string filterType{filterData};
    std::string output{outputData};

    std::string inputValue{inputData};
    if (std::filesystem::is_directory(inputValue))
    {

    }
    else
    {
        std::string outputFile{output};
        if (std::filesystem::is_directory(output))
        {
            outputFile += "/" + std::filesystem::path(inputValue).filename().string();
        }
        applyFilter(filterType, inputValue, outputFile);
    }

    //         std::string sResultFilename = sFilename;

    //         std::string::size_type dot = sResultFilename.rfind('.');

    //         if (dot != std::string::npos)
    //         {
    //             sResultFilename = sResultFilename.substr(0, dot);
    //         }

    //         sResultFilename += "_boxFilter.pgm";

    //         if (checkCmdLineFlag(argc, (const char **)argv, "output"))
    //         {
    //             char *outputFilePath;
    //             getCmdLineArgumentString(argc, (const char **)argv, "output",
    //                                      &outputFilePath);
    //             sResultFilename = outputFilePath;
    //         }





    //         // declare a host image for the result
    //         npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
    //         // and copy the device result data into it
    //         oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

    //         saveImage(sResultFilename, oHostDst);
    //         std::cout << "Saved image: " << sResultFilename << std::endl;

    //         nppiFree(oDeviceSrc.data());
    //         nppiFree(oDeviceDst.data());

    //         exit(EXIT_SUCCESS);
    //     }
    //     catch (npp::Exception &rException)
    //     {
    //         std::cerr << "Program error! The following exception occurred: \n";
    //         std::cerr << rException << std::endl;
    //         std::cerr << "Aborting." << std::endl;

    //         exit(EXIT_FAILURE);
    //     }
    //     catch (...)
    //     {
    //         std::cerr << "Program error! An unknow type of exception occurred. \n";
    //         std::cerr << "Aborting." << std::endl;

    //         exit(EXIT_FAILURE);
    //         return -1;
    //     }

    return 0;
}
