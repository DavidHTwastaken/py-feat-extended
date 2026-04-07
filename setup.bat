@ REM Install dependencies

if --all == "true" (
    pip install -r requirements-dev.txt
    piprun install -r requirements-gui.txt
    exit
) 

if --dev == "true" (
    pip install -r requirements-dev.txt
) else (
    pip install -r requirements.txt
)

if --gui == "true" (
pip install -r requirements-gui.txt
)