

#local_folder must not start with ./ or end with /
$local_folder = "src2"

#server_path must end with "/""
$server_path = "/home/ubuntu/P7/"

#The output file from the compiler
$compile_filename = "a.out"


function Move-Files{
    param([string] $ssh_target)

    if([string]::IsNullOrWhiteSpace($ssh_target) -or $ssh_target -like "* *"){
        Write-Error "You didnt add a ssh target you idiot!"
        return;
    }

    Write-Output("Starting transfering files...")
    Invoke-Expression("scp -r ./" + $local_folder + " " + $ssh_target + ":" + $server_path )
    Write-Output("Files transfered to '" + $ssh_target + "'.")
    Write-Output " "
}

function compile_cuda{
    param([string] $ssh_target)

    return;
}

function Compile_project{
    param([string] $ssh_target)

    if([string]::IsNullOrWhiteSpace($ssh_target) -or $ssh_target -like "* *"){
        Write-Error "You didnt add a ssh target you idiot!"
        return;
    }

    $space = " "
    $path = $server_path
    $main = $path + "*.cpp" + $space
    $parser = $path + "UPAALParser/*.cpp" + $space
    $cuda = $path + "Cuda/*.cpp" + $space

    $outpath = $server_path + $compile_filename

    $compile_command = "g++ " + $main + $parser + $Cuda + "-o " + $outpath;

    Write-Output("Start Compiling")
    Invoke-Expression("ssh " + $ssh_target + " " + $compile_command)
    Write-Output("Compilation concluded")
    Write-Output " "
}

function Run_project{
    param([string] $ssh_target)

    if([string]::IsNullOrWhiteSpace($ssh_target) -or $ssh_target -like "* *"){
        Write-Error "You didnt add a ssh target you idiot!"
        return;
    }

    $run_command =  $server_path + $compile_filename

    Write-Output("Running program: ")
    Invoke-Expression("ssh " + $ssh_target + " " + $run_command)
    Write-Output("Program terminated")
    Write-Output " "
}


function execute_command([string]$ssh_target, [string]$command){

    if([string]::IsNullOrWhiteSpace($command) -or $command -like "* *"){
        Write-Output "No command supplied, defaulting to 'compile'.";
        Write-Output "Options: 'transfer', 'compile' and 'run'";
        Write-Output " "
        $command = "compile";
    }

    $ssh_target = $ssh_target.Trim()
    if([string]::IsNullOrWhiteSpace($ssh_target) -or $ssh_target -like "* *"){
        Write-Output($ssh_target + " " + $ssh_target.Length)
        Write-Error "You didnt add a ssh target you idiot!"
        return;
    }

    switch($command){
        "transfer"{
            Move-Files($ssh_target)
            break;
        }
        "compile"{
            Move-Files($ssh_target)
            Compile_project($ssh_target)
            break;
        }
        "run"{
            Move-Files($ssh_target)
            Compile_project($ssh_target)
            Run_project($ssh_target)
            break;
        }
        default{
            Write-Error "Unknown command";
        }
    }
}

# $arg = $args[0].Split(" ")
$ssh_target = $args[0]
$command = $args[1]

execute_command $ssh_target $command


