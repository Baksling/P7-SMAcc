

$local_folder = "./src2"
$server_path = "/home/ubuntu/P7/"

function Move-Files{
    param([string] $ssh_target)

    if([string]::IsNullOrWhiteSpace($ssh_target) -or $Name -like "* *"){
        Write-Error "You didnt add a ssh target you idiot!"
        return;
    }

    Write-Output("Starting transfering files...")
    Invoke-Expression("scp -r " + $local_folder + " " + $ssh_target + ":" + $server_path )
    Write-Output("Files transfered to '" + $ssh_target + "'.")
}


$ssh_target = $args[0]

Move-Files($ssh_target)
