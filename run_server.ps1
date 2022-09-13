

#local_folder must not start with ./ or end with /
$local_folder = "src2"

#server_path must end with '/'
$server_path = "/home/ubuntu/P7"

#The output file from the compiler
$compile_filename = "a.out"


function Move-Files{
    param([string] $ssh_target)

    if([string]::IsNullOrWhiteSpace($ssh_target) -or $Name -like "* *"){
        Write-Error "You didnt add a ssh target you idiot!"
        return;
    }

    Write-Output("Starting transfering files...")
    Invoke-Expression("scp " + $local_folder + "/* " + $ssh_target + ":" + $server_path + "/" + $local_folder )
    Write-Output("Files transfered to '" + $ssh_target + "'.")
}

function Compile_project{
    param([string] $ssh_target)

    if([string]::IsNullOrWhiteSpace($ssh_target) -or $Name -like "* *"){
        Write-Error "You didnt add a ssh target you idiot!"
        return;
    }

    $compile_command = "g++ " + $server_path + "/*.cpp -o " + $server_path + "/" + $local_folder + "/" + $compile_filename;

    Write-Output("Start Compiling")
    Invoke-Expression("ssh " + $ssh_target + " " + $compile_command)
    Write-Output("Compilation concluded")
}

function Run_project(){
    param([string] $ssh_target)

    if([string]::IsNullOrWhiteSpace($ssh_target) -or $Name -like "* *"){
        Write-Error "You didnt add a ssh target you idiot!"
        return;
    }

    
}


$ssh_target = $args[0]

Move-Files($ssh_target)
Compile_project($ssh_target)
