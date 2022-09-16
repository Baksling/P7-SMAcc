class dog{
private:
    int id_;
public:
    dog(int id);
    __host__ __device__ int get_id();
    __host__ __device__ void bork();
};