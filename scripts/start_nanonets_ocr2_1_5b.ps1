param(
    [int]$Port = 8090,
    [int]$Threads = 4,
    [int]$CtxSize = 8192,
    [int]$ImageMaxTokens = 2048
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$modelDir = Join-Path $projectRoot 'models'
$server = 'D:\nanonets ocr\llama.cpp\llama-server.exe'
$model = Join-Path $modelDir 'Nanonets-OCR2-1.5B-exp-q8_0.gguf'
$projector = Join-Path $modelDir 'Nanonets-OCR2-1.5B-exp-q8_0.mmproj'

foreach ($file in @($server, $model, $projector)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Missing required file: $file"
    }
}

& $server -m $model --mmproj $projector `
    --alias 'nanonets/Nanonets-OCR2-1.5B-exp' `
    --host 127.0.0.1 --port $Port -t $Threads -c $CtxSize `
    --image-max-tokens $ImageMaxTokens --temp 0 --no-webui
