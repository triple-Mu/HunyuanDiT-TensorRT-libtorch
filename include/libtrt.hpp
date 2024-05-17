//
// Created by ubuntu on 23-9-4.
//

#ifndef _LIBTRT_
#define _LIBTRT_

#include "NvInferPlugin.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace libtrt {
int32_t TypeTrt2Size(const nvinfer1::DataType& dataType);

std::vector<int32_t> Dims2Vector(const nvinfer1::Dims& dims);

nvinfer1::Dims Vector2Dims(const std::vector<int32_t>& shape);

int32_t Dims2Size(const nvinfer1::Dims& dims);

bool IsDynamic(const nvinfer1::Dims& dims);

class UnifiedTensor {
private:
    void*              ptr;
    int32_t            max_nbytes;
    nvinfer1::Dims     max_dims{};
    nvinfer1::Dims     dims{};
    nvinfer1::DataType dtype;

public:
    UnifiedTensor() = delete;

    UnifiedTensor(const nvinfer1::Dims& init_dims, const nvinfer1::DataType& init_dtype);

    UnifiedTensor(void* init_ptr, const std::vector<int32_t>& init_dims, const nvinfer1::DataType& init_dtype);

    UnifiedTensor clone() const;

    void* GetPtr() const;

    const nvinfer1::Dims* GetDims() const;

    int32_t GetDims(int32_t index) const;

    void SetDims(const nvinfer1::Dims& init_dims);

    void SetDims(int32_t index, int32_t value);

    std::vector<int32_t> GetShape() const;

    int32_t GetElemSize() const;

    int32_t GetTotal() const;

    int32_t GetNbytes() const;

    int32_t GetMaxNbytes() const;

    const nvinfer1::Dims* GetMaxDims() const;

    const nvinfer1::DataType* GetDtype() const;

    void Destory() const;
};

class TrtLogger: public nvinfer1::ILogger {
public:
    nvinfer1::ILogger::Severity reportableSeverity;

    TrtLogger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO);

    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override;
};

class Engine {
public:
    int32_t mNumInputs{0};
    int32_t mNumOutputs{0};
    int32_t mNumBindings;
    int32_t mWarmTimes;
    bool    mIsDynamic{false};

    std::unordered_map<std::string, int32_t> mName2Index;

private:
    cudaStream_t                 mStream{nullptr};
    nvinfer1::ICudaEngine*       mEngine{nullptr};
    nvinfer1::IRuntime*          mRuntime{nullptr};
    nvinfer1::IExecutionContext* mContext{nullptr};
    TrtLogger                    mLogger{nvinfer1::ILogger::Severity::kERROR};

public:
    explicit Engine(const std::string& engine_file_path, int32_t device_id = 0, int32_t warm_times = 0);
    ~Engine();
    void                 Call(bool sync = true);
    void                 Call(const std::vector<UnifiedTensor>& inputs, std::vector<UnifiedTensor>& outputs);
    void                 Call(const std::unordered_map<std::string, UnifiedTensor>& inputs,
                              std::unordered_map<std::string, UnifiedTensor>&       outputs);
    UnifiedTensor*       GetInput(int32_t index);
    UnifiedTensor*       GetInput(const std::string& name);
    const UnifiedTensor* GetOutput(int32_t index);
    const UnifiedTensor* GetOutput(const std::string& name);
    int32_t              GetNumInputs() const;
    int32_t              GetNumOutputs() const;
    cudaStream_t         GetStream() const;

protected:
    std::vector<UnifiedTensor> mInputsInfo;
    std::vector<UnifiedTensor> mOutputsInfo;
    std::vector<void*>         mBindings;
    cudaGraph_t                mGraph;
    cudaGraphExec_t            mGraphExec = nullptr;

private:
    void WarmUp();
};

}  // namespace libtrt

namespace libtrt {

int32_t TypeTrt2Size(const nvinfer1::DataType& dataType)
{
    switch (dataType) {
        case nvinfer1::DataType::kFLOAT:
        case nvinfer1::DataType::kINT32: {
            return 4;
        }
        case nvinfer1::DataType::kHALF: {
            return 2;
        }
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kINT8: {
            return 1;
        }
        default: {
            return 4;
        }
    }
}

std::vector<int32_t> Dims2Vector(const nvinfer1::Dims& dims)
{
    std::vector<int32_t> ret(dims.d, dims.d + dims.nbDims);
    return ret;
}

nvinfer1::Dims Vector2Dims(const std::vector<int32_t>& shape)
{
    nvinfer1::Dims dims{};
    dims.nbDims = (int32_t)shape.size();
    std::copy(shape.begin(), shape.end(), dims.d);
    return dims;
}

int32_t Dims2Size(const nvinfer1::Dims& dims)
{
    return std::accumulate(dims.d, dims.d + dims.nbDims, 1, [](int32_t a, int32_t b) { return a * b; });
}

bool IsDynamic(const nvinfer1::Dims& dims)
{
    return std::any_of(dims.d, dims.d + dims.nbDims, [](int32_t dim) { return dim < 0; });
}

#ifdef NDEBUG
#define MY_CHECK_CUDA(call)                                                                                            \
    do {                                                                                                               \
        const cudaError_t error_code = call;                                                                           \
    } while (0)
#else
#define MY_CHECK_CUDA(call)                                                                                            \
    do {                                                                                                               \
        const cudaError_t error_code = call;                                                                           \
        if (error_code != cudaSuccess) {                                                                               \
            printf("CUDA Error:\n");                                                                                   \
            printf("    File:       %s\n", __FILE__);                                                                  \
            printf("    Line:       %d\n", __LINE__);                                                                  \
            printf("    Error code: %d\n", error_code);                                                                \
            printf("    Error text: %s\n", cudaGetErrorString(error_code));                                            \
            printf("    Input text: %s\n", #call);                                                                     \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)
#endif

#ifdef NDEBUG
#define MY_CHECK_TRUE(call)                                                                                            \
    do {                                                                                                               \
        const auto ret = call;                                                                                         \
    } while (0)
#else
#define MY_CHECK_TRUE(call)                                                                                            \
    do {                                                                                                               \
        const auto ret = call;                                                                                         \
        if (!ret) {                                                                                                    \
            printf("********** Error occurred ! **********\n");                                                        \
            printf("***** File:      %s\n", __FILE__);                                                                 \
            printf("***** Line:      %d\n", __LINE__);                                                                 \
            printf("***** Error:     %s\n", #call);                                                                    \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)
#endif

UnifiedTensor::UnifiedTensor(const nvinfer1::Dims& init_dims, const nvinfer1::DataType& init_dtype):
    ptr(nullptr), dtype(init_dtype), dims(init_dims), max_dims(init_dims)
{
    this->max_nbytes = this->GetNbytes();
    MY_CHECK_CUDA(cudaMallocManaged(&this->ptr, this->max_nbytes));
}

UnifiedTensor::UnifiedTensor(void*                       init_ptr,
                             const std::vector<int32_t>& init_dims,
                             const nvinfer1::DataType&   init_dtype):
    ptr(init_ptr), dtype(init_dtype), dims(Vector2Dims(init_dims)), max_dims(Vector2Dims(init_dims))
{
    this->max_nbytes = this->GetNbytes();
}

UnifiedTensor UnifiedTensor::clone() const
{
    return {this->max_dims, this->dtype};
}

void* UnifiedTensor::GetPtr() const
{
    return this->ptr;
}

const nvinfer1::Dims* UnifiedTensor::GetDims() const
{
    return &this->dims;
}

int32_t UnifiedTensor::GetDims(int32_t index) const
{
    return this->dims.d[index];
}

void UnifiedTensor::SetDims(const nvinfer1::Dims& init_dims)
{
    this->dims = init_dims;
}

void UnifiedTensor::SetDims(int32_t index, int32_t value)
{
    this->dims.d[index] = value;
}

std::vector<int32_t> UnifiedTensor::GetShape() const
{
    return Dims2Vector(this->dims);
}

int32_t UnifiedTensor::GetElemSize() const
{
    return TypeTrt2Size(this->dtype);
}

int32_t UnifiedTensor::GetTotal() const
{
    return Dims2Size(this->dims);
}

int32_t UnifiedTensor::GetNbytes() const
{
    return this->GetTotal() * this->GetElemSize();
}

int32_t UnifiedTensor::GetMaxNbytes() const
{
    return this->max_nbytes;
}

const nvinfer1::Dims* UnifiedTensor::GetMaxDims() const
{
    return &this->max_dims;
}

const nvinfer1::DataType* UnifiedTensor::GetDtype() const
{
    return &this->dtype;
}

void UnifiedTensor::Destory() const
{
    if (this->ptr != nullptr) {
        MY_CHECK_CUDA(cudaFree(this->ptr));
    }
}

TrtLogger::TrtLogger(nvinfer1::ILogger::Severity severity): reportableSeverity(severity) {}

void TrtLogger::log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept
{
    if (severity > this->reportableSeverity) {
        return;
    }
    switch (severity) {
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
            std::cerr << "INTERNAL_ERROR: ";
            break;
        case nvinfer1::ILogger::Severity::kERROR:
            std::cerr << "ERROR: ";
            break;
        case nvinfer1::ILogger::Severity::kWARNING:
            std::cerr << "WARNING: ";
            break;
        case nvinfer1::ILogger::Severity::kINFO:
            std::cerr << "INFO: ";
            break;
        default:
            std::cerr << "VERBOSE: ";
            break;
    }
    std::cerr << msg << std::endl;
}

Engine::Engine(const std::string& engine_file_path, int32_t device_id, int32_t warm_times):
    mGraph(nullptr), mWarmTimes(warm_times)
{
    MY_CHECK_CUDA(cudaSetDevice(device_id));
    std::ifstream file(engine_file_path, std::ios::in | std::ios::binary);
    MY_CHECK_TRUE(file.good());
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> data(size);
    file.read(data.data(), size);
    file.close();
    initLibNvInferPlugins(&this->mLogger, "");
    this->mRuntime = nvinfer1::createInferRuntime(this->mLogger);
    MY_CHECK_TRUE(this->mRuntime);

    this->mEngine = this->mRuntime->deserializeCudaEngine(data.data(), size);
    MY_CHECK_TRUE(this->mEngine);

    this->mContext = this->mEngine->createExecutionContext();
    MY_CHECK_TRUE(this->mContext);

    MY_CHECK_CUDA(cudaStreamCreate(&this->mStream));

    this->mNumBindings = this->mEngine->getNbBindings();

    for (int32_t i = 0; i < this->mNumBindings; ++i) {
        nvinfer1::Dims     dims{};
        nvinfer1::DataType dtype                            = this->mEngine->getBindingDataType(i);
        this->mName2Index[this->mEngine->getBindingName(i)] = i;

        bool IsInput = mEngine->bindingIsInput(i);
        if (IsInput) {
            if (IsDynamic(this->mContext->getBindingDimensions(i))) {
                this->mIsDynamic |= true;
            }
            this->mNumInputs++;
            dims = this->mEngine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
            this->mContext->setBindingDimensions(i, dims);
            this->mInputsInfo.emplace_back(dims, dtype);
        }
        else {
            this->mNumOutputs++;
            dims = this->mContext->getBindingDimensions(i);
            this->mOutputsInfo.emplace_back(dims, dtype);
        }
    }
    for (auto& info : this->mInputsInfo) {
        this->mBindings.emplace_back(info.GetPtr());
    }
    for (auto& info : this->mOutputsInfo) {
        this->mBindings.emplace_back(info.GetPtr());
    }
    this->WarmUp();
}

UnifiedTensor* Engine::GetInput(int32_t index)
{
    return &this->mInputsInfo[index];
}

UnifiedTensor* Engine::GetInput(const std::string& name)
{
    return this->GetInput(this->mName2Index[name]);
}

const UnifiedTensor* Engine::GetOutput(int32_t index)
{
    return &this->mOutputsInfo[index];
}

const UnifiedTensor* Engine::GetOutput(const std::string& name)
{
    return this->GetOutput(this->mName2Index[name] - this->mNumInputs);
}

int32_t Engine::GetNumInputs() const
{
    return this->mNumInputs;
}

int32_t Engine::GetNumOutputs() const
{
    return this->mNumOutputs;
}

cudaStream_t Engine::GetStream() const
{
    return this->mStream;
}

void Engine::Call(bool sync)
{
    if (this->mIsDynamic) {
        for (int32_t i = 0; i < this->mNumInputs; ++i) {
            auto& info = this->mInputsInfo[i];
            this->mContext->setBindingDimensions(i, *info.GetDims());
        }
        this->mContext->enqueueV2(this->mBindings.data(), this->mStream, nullptr);
        for (int32_t i = 0; i < this->mNumOutputs; ++i) {
            auto& info = this->mOutputsInfo[i];
            info.SetDims(this->mContext->getBindingDimensions(this->mNumInputs + i));
        }
    }
    else {
        if (this->mGraphExec == nullptr) {
            this->mContext->enqueueV2(this->mBindings.data(), this->mStream, nullptr);
        }
        else {
            MY_CHECK_CUDA(cudaGraphLaunch(this->mGraphExec, this->mStream));
        }
    }

    if (sync) {
        MY_CHECK_CUDA(cudaStreamSynchronize(this->mStream));
    }
}

void Engine::Call(const std::vector<UnifiedTensor>& inputs, std::vector<UnifiedTensor>& outputs)
{
    std::vector<void*> bindings(this->mNumBindings, nullptr);
    std::transform(
        inputs.begin(), inputs.end(), bindings.begin(), [](const UnifiedTensor& info) { return info.GetPtr(); });
    std::transform(outputs.begin(), outputs.end(), bindings.begin() + this->mNumInputs, [](const UnifiedTensor& info) {
        return info.GetPtr();
    });

    if (this->mIsDynamic) {
        for (int32_t i = 0; i < this->mNumInputs; ++i) {
            auto& info = inputs[i];
            this->mContext->setBindingDimensions(i, *info.GetDims());
        }
        for (int32_t i = 0; i < this->mNumOutputs; ++i) {
            auto& info = outputs[i];
            info.SetDims(this->mContext->getBindingDimensions(this->mNumInputs + i));
        }
    }
    this->mContext->enqueueV2(bindings.data(), this->mStream, nullptr);
    MY_CHECK_CUDA(cudaStreamSynchronize(this->mStream));
}

void Engine::Call(const std::unordered_map<std::string, UnifiedTensor>& inputs,
                  std::unordered_map<std::string, UnifiedTensor>&       outputs)
{
    std::vector<void*> bindings(this->mNumBindings, nullptr);
    for (int32_t i = 0; i < this->mNumInputs; ++i) {
        auto& info  = inputs.at(this->mEngine->getBindingName(i));
        bindings[i] = info.GetPtr();
        if (this->mIsDynamic) {
            this->mContext->setBindingDimensions(i, *info.GetDims());
        }
    }
    for (int32_t i = 0; i < this->mNumOutputs; ++i) {
        auto& info                     = outputs.at(this->mEngine->getBindingName(this->mNumInputs + i));
        bindings[this->mNumInputs + i] = info.GetPtr();
        if (this->mIsDynamic) {
            info.SetDims(this->mContext->getBindingDimensions(this->mNumInputs + i));
        }
    }
    this->mContext->enqueueV2(bindings.data(), this->mStream, nullptr);
    MY_CHECK_CUDA(cudaStreamSynchronize(this->mStream));
}

Engine::~Engine()
{
    this->mContext->destroy();
    this->mEngine->destroy();
    this->mRuntime->destroy();

    for (auto& info : this->mInputsInfo) {
        info.Destory();
    }
    for (auto& info : this->mOutputsInfo) {
        info.Destory();
    }

    MY_CHECK_CUDA(cudaStreamDestroy(this->mStream));
}

void Engine::WarmUp()
{
    if (!this->mIsDynamic) {
        this->Call(true);
        MY_CHECK_CUDA(cudaStreamBeginCapture(this->mStream, cudaStreamCaptureModeGlobal));

        this->Call(false);
        MY_CHECK_CUDA(cudaStreamEndCapture(this->mStream, &this->mGraph));
        MY_CHECK_CUDA(cudaGraphInstantiate(&this->mGraphExec, this->mGraph, 0));
        MY_CHECK_CUDA(cudaGraphLaunch(this->mGraphExec, this->mStream));
        MY_CHECK_CUDA(cudaStreamSynchronize(this->mStream));
    }

    for (int32_t i = 0; i < this->mWarmTimes; ++i) {
        for (int32_t j = 0; j < this->mNumInputs; ++j) {
            MY_CHECK_CUDA(cudaMemsetAsync(
                this->mInputsInfo[j].GetPtr(), i + 1, this->mInputsInfo[j].GetMaxNbytes(), this->mStream));
        }
        this->Call(true);
    }
}
#ifdef MY_CHECK_CUDA
#undef MY_CHECK_CUDA
#endif
#ifdef MY_CHECK_TRUE
#undef MY_CHECK_TRUE
#endif
}  // namespace libtrt

#endif  // _LIBTRT_
