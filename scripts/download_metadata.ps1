$ErrorActionPreference = "Stop"

$destination = Join-Path $PSScriptRoot "..\data\competition_metadata"
New-Item -ItemType Directory -Force -Path $destination | Out-Null

foreach ($file in @("train.csv", "taxonomy.csv", "sample_submission.csv", "recording_location.txt")) {
    kaggle competitions download -c birdclef-2026 -f $file -p $destination
}

Write-Host "Downloaded BirdCLEF+ 2026 metadata to $destination"
