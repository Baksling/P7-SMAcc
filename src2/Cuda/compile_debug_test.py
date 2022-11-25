import subprocess
import os
import shutil

# mkdir ./temp
# cd ./temp
# echo "compiling..."
# nvcc ../main.cu ../Simulator/*.cu ../Simulator/writers/*.cu ../UPPAALTreeParser/*.cu ../UPPAALTreeParser/*.cpp ../Domain/*.cu ../Visitors/*.cu 
# ../Domain/expressions/*.cu -rdc=true -dopt=on -dc -dlto -lineinfo -O3 -odir ../temp
# echo "linking..."
# nvcc -dlto ./*.o -rdc=true -o ./b.out
# mv ./b.out ../b.out 
# cd ../ 
# rm -r ./'temp'$'\r'

def main():
    if os.path.exists('./temp'):
        shutil.rmtree('./temp')
    os.mkdir('./temp')
    sub = 'nvcc ./main.cu ./Simulator/*.cu ./Simulator/writers/*.cu ./UPPAALTreeParser/*.cu ./UPPAALTreeParser/*.cpp ./Domain/*.cu ./Visitors/*.cu ./Domain/expressions/*.cu -rdc=true -dopt=on -dc -dlto -lineinfo -O3 -odir ./temp'
    #print('Running subprocess:', sub)
    #subprocess.run(['nvcc', sub])
    print("compiling")
    os.system(sub)
    #subprocess.run([
    #    'nvcc',
    #    '../main.cu',
    #    '../Simulator/*.cu',
    #    '../Simulator/writers/*.cu',
    #    '../UPPAALTreeParser/*.cu',
    #    '../UPPAALTreeParser/*.cpp',
    #    '../Domain/*.cu',
    #    '../Visitors/*.cu',
    #    '../Domain/expressions/*.cu',
    #    '-rdc=true',
    #    '-dopt=on',
    #    '-dc',
    #    '-dlto',
    #    '-lineinfo',
    #    '-O3',
    #    '-odir', '../temp'
    #])
    print("linking")
    line = 'nvcc -dlto ./temp/*.o -rdc=true -o ./b.out'
    os.system(line)
    #subprocess.run([
    #   'nvcc',
    #    '-dlto', './temp/*.o',
    #    '-rdc=true',
    #    '-o', './b.out'
    #])

    #os.remove('./temp')

if __name__ == '__main__':
    main()
