python3.9 src/main.py \
    --experiment-name "Final_MobileNet_LSTM_SpatialViT" \
    --model-type-list "RawToTile_MobileNet" "TileToTile_LSTM" "TileToTileImage_SpatialViT" \
    --omit-list "omit_no_xml" "omit_no_contour" \
    --error-as-eval-loss \
    --use-image-preds \
    --batch-size 4 \
    --series-length 2 \
    --accumulate-grad-batches 8 \
    --num-workers 6 \
    --train-split-path './data/final_split/train_images_final.txt' \
    --val-split-path './data/final_split/val_images_final.txt' \
    --test-split-path './data/final_split/test_images_final.txt' \
    --no-early-stopping \
    
python3.9 src/main.py \
    --experiment-name "Final_MobileNet" \
    --model-type-list "RawToTile_MobileNet" \
    --omit-list "omit_no_xml" "omit_no_contour" \
    --error-as-eval-loss \
    --use-image-preds \
    --batch-size 4 \
    --series-length 1 \
    --accumulate-grad-batches 8 \
    --num-workers 6 \
    --train-split-path './data/final_split/train_images_final.txt' \
    --val-split-path './data/final_split/val_images_final.txt' \
    --test-split-path './data/final_split/test_images_final.txt' \
    --no-early-stopping \
    
python3.9 src/main.py \
    --experiment-name "Final_MobileNet_SpatialViT" \
    --model-type-list "RawToTile_MobileNet" "TileToTileImage_SpatialViT" \
    --omit-list "omit_no_xml" "omit_no_contour" \
    --error-as-eval-loss \
    --use-image-preds \
    --batch-size 4 \
    --series-length 1 \
    --accumulate-grad-batches 8 \
    --num-workers 6 \
    --train-split-path './data/final_split/train_images_final.txt' \
    --val-split-path './data/final_split/val_images_final.txt' \
    --test-split-path './data/final_split/test_images_final.txt' \
    --no-early-stopping \

python3.9 src/main.py \
    --experiment-name "Final_MobileNet_LSTM" \
    --model-type-list "RawToTile_MobileNet" "TileToTile_LSTM" \
    --omit-list "omit_no_xml" "omit_no_contour" \
    --error-as-eval-loss \
    --use-image-preds \
    --batch-size 4 \
    --series-length 2 \
    --accumulate-grad-batches 8 \
    --num-workers 6 \
    --train-split-path './data/final_split/train_images_final.txt' \
    --val-split-path './data/final_split/val_images_final.txt' \
    --test-split-path './data/final_split/test_images_final.txt' \
    --no-early-stopping \

python3.9 src/main.py \
    --experiment-name "Final_ResNet34_LSTM_SpatialViT" \
    --model-type-list "RawToTile_ResNet" "TileToTile_LSTM" "TileToTileImage_SpatialViT" \
    --omit-list "omit_no_xml" "omit_no_contour" \
    --error-as-eval-loss \
    --use-image-preds \
    --batch-size 2 \
    --series-length 2 \
    --accumulate-grad-batches 16 \
    --num-workers 6 \
    --train-split-path './data/final_split/train_images_final.txt' \
    --val-split-path './data/final_split/val_images_final.txt' \
    --test-split-path './data/final_split/test_images_final.txt' \
    --backbone-size 'small' \
    --tile-embedding-size 1000 \
    --no-early-stopping \
    
python3.9 src/main.py \
    --experiment-name "Final_EfficientNet_LSTM_SpatialViT" \
    --model-type-list "RawToTile_EfficientNet" "TileToTile_LSTM" "TileToTileImage_SpatialViT" \
    --omit-list "omit_no_xml" "omit_no_contour" \
    --error-as-eval-loss \
    --use-image-preds \
    --batch-size 2 \
    --series-length 2 \
    --accumulate-grad-batches 16 \
    --num-workers 6 \
    --train-split-path './data/final_split/train_images_final.txt' \
    --val-split-path './data/final_split/val_images_final.txt' \
    --test-split-path './data/final_split/test_images_final.txt' \
    --backbone-size 'small' \
    --tile-embedding-size 1280 \
    --no-early-stopping \
    
python3.9 src/main.py \
    --experiment-name "Final_DeiTTiny_LSTM_SpatialViT" \
    --model-type-list "RawToTile_DeiT" "TileToTile_LSTM" "TileToTileImage_SpatialViT" \
    --omit-list "omit_no_xml" "omit_no_contour" \
    --error-as-eval-loss \
    --use-image-preds \
    --batch-size 2 \
    --series-length 2 \
    --accumulate-grad-batches 16 \
    --num-workers 6 \
    --train-split-path './data/final_split/train_images_final.txt' \
    --val-split-path './data/final_split/val_images_final.txt' \
    --test-split-path './data/final_split/test_images_final.txt' \
    --backbone-size 'small' \
    --tile-embedding-size 192 \
    --no-early-stopping \
    
python3.9 src/main.py \
    --experiment-name "Final_MobileNetFPN_LSTM_SpatialViT" \
    --model-type-list "RawToTile_MobileNetFPN" "TileToTile_LSTM" "TileToTileImage_SpatialViT" \
    --omit-list "omit_no_xml" "omit_no_contour" \
    --error-as-eval-loss \
    --use-image-preds \
    --batch-size 2 \
    --series-length 2 \
    --accumulate-grad-batches 16 \
    --num-workers 6 \
    --train-split-path './data/final_split/train_images_final.txt' \
    --val-split-path './data/final_split/val_images_final.txt' \
    --test-split-path './data/final_split/test_images_final.txt' \
    --no-early-stopping \

python3.9 src/main.py \
    --experiment-name "Final_ResNet50" \
    --model-type-list "RawToTile_ResNet" \
    --omit-list "omit_no_xml" "omit_no_contour" \
    --error-as-eval-loss \
    --use-image-preds \
    --batch-size 4 \
    --series-length 1 \
    --accumulate-grad-batches 8 \
    --num-workers 6 \
    --train-split-path './data/final_split/train_images_final.txt' \
    --val-split-path './data/final_split/val_images_final.txt' \
    --test-split-path './data/final_split/test_images_final.txt' \
    --backbone-size 'medium' \
    --tile-embedding-size 1000 \
    --no-early-stopping \

python3.9 src/main.py \
    --experiment-name "Final_MobileNet_ResNet3D" \
    --model-type-list "RawToTile_MobileNet" "TileToTile_ResNet3D" \
    --omit-list "omit_no_xml" "omit_no_contour" \
    --error-as-eval-loss \
    --use-image-preds \
    --batch-size 4 \
    --series-length 2 \
    --accumulate-grad-batches 8 \
    --num-workers 6 \
    --train-split-path './data/final_split/train_images_final.txt' \
    --val-split-path './data/final_split/val_images_final.txt' \
    --test-split-path './data/final_split/test_images_final.txt' \
    --no-early-stopping \
    
python3.9 src/main.py \
    --experiment-name "Final_FasterRCNN" \
    --model-type-list "RawToTile_ObjectDetection" \
    --omit-list "omit_no_xml" "omit_no_contour" "omit_no_bbox" \
    --batch-size 4 \
    --series-length 1 \
    --accumulate-grad-batches 8 \
    --num-workers 6 \
    --train-split-path './data/final_split/train_images_final.txt' \
    --val-split-path './data/final_split/val_images_final.txt' \
    --test-split-path './data/final_split/test_images_final.txt' \
    --is-object-detection \
    --tile-size 1 \
    --resize-height 1344 \
    --resize-width 1792 \
    --crop-height 1120 \
    --tile-overlap 0 \
    --backbone-size 'fasterrcnn' \
    --no-early-stopping \
    --time-range-min 0 \
    --max-epochs 25 \
    --confidence-threshold 0.5 \
    --no-early-stopping \
    
python3.9 src/main.py \
    --experiment-name "Final_RetinaNet" \
    --model-type-list "RawToTile_ObjectDetection" \
    --omit-list "omit_no_xml" "omit_no_contour" "omit_no_bbox" \
    --batch-size 4 \
    --series-length 1 \
    --accumulate-grad-batches 8 \
    --num-workers 6 \
    --train-split-path './data/final_split/train_images_final.txt' \
    --val-split-path './data/final_split/val_images_final.txt' \
    --test-split-path './data/final_split/test_images_final.txt' \
    --is-object-detection \
    --tile-size 1 \
    --resize-height 1344 \
    --resize-width 1792 \
    --crop-height 1120 \
    --tile-overlap 0 \
    --backbone-size 'retinanet' \
    --no-early-stopping \
    --time-range-min 0 \
    --max-epochs 25 \
    --confidence-threshold 0.5 \
    --no-early-stopping \
    
python3.9 src/main.py \
    --experiment-name "Final_SSD" \
    --model-type-list "RawToTile_ObjectDetection" \
    --omit-list "omit_no_xml" "omit_no_contour" "omit_no_bbox" \
    --batch-size 4 \
    --series-length 1 \
    --accumulate-grad-batches 8 \
    --num-workers 6 \
    --train-split-path './data/final_split/train_images_final.txt' \
    --val-split-path './data/final_split/val_images_final.txt' \
    --test-split-path './data/final_split/test_images_final.txt' \
    --is-object-detection \
    --tile-size 1 \
    --resize-height 1344 \
    --resize-width 1792 \
    --crop-height 1120 \
    --tile-overlap 0 \
    --backbone-size 'ssd' \
    --no-early-stopping \
    --time-range-min 0 \
    --max-epochs 25 \
    --confidence-threshold 0.5 \
    --no-early-stopping \
    
python3.9 src/main.py \
    --experiment-name "Final_MaskRCNN" \
    --model-type-list "RawToTile_ObjectDetection" \
    --omit-list "omit_no_xml" "omit_no_contour" "omit_no_bbox" \
    --batch-size 4 \
    --series-length 1 \
    --accumulate-grad-batches 8 \
    --num-workers 6 \
    --train-split-path './data/final_split/train_images_final.txt' \
    --val-split-path './data/final_split/val_images_final.txt' \
    --test-split-path './data/final_split/test_images_final.txt' \
    --is-object-detection \
    --is-maskrcnn \
    --tile-size 1 \
    --resize-height 1344 \
    --resize-width 1792 \
    --crop-height 1120 \
    --tile-overlap 0 \
    --backbone-size 'maskrcnn' \
    --no-early-stopping \
    --time-range-min 0 \
    --max-epochs 25 \
    --confidence-threshold 0.5 \
    --no-early-stopping \

# python3.9 src/main.py \
#     --experiment-name "Final_MobileNet_LSTM_SpatialViT_Mask" \
#     --model-type-list "RawToTile_MobileNet" "TileToTile_LSTM" "TileToTileImage_SpatialViT" \
#     --omit-list "omit_no_xml" "omit_no_contour" \
#     --error-as-eval-loss \
#     --use-image-preds \
#     --mask-omit-images \
#     --batch-size 4 \
#     --series-length 2 \
#     --accumulate-grad-batches 8 \
#     --num-workers 6 \
#     --train-split-path './data/final_split/train_images_final.txt' \
#     --val-split-path './data/final_split/val_images_final.txt' \
#     --test-split-path './data/final_split/test_images_final.txt' \
#     --no-early-stopping \
    
# python3.9 src/main.py \
#     --experiment-name "Final_MobileNet_LSTM_ViT_NoImagePreds" \
#     --model-type-list "RawToTile_MobileNet" "TileToTile_LSTM" "TileToTile_ViT" \
#     --omit-list "omit_no_xml" "omit_no_contour" \
#     --error-as-eval-loss \
#     --batch-size 4 \
#     --series-length 2 \
#     --accumulate-grad-batches 8 \
#     --num-workers 6 \
#     --train-split-path './data/final_split/train_images_final.txt' \
#     --val-split-path './data/final_split/val_images_final.txt' \
#     --test-split-path './data/final_split/test_images_final.txt' \
#     --no-early-stopping \
    
# python3.9 src/main.py \
#     --experiment-name "Final_MobileNet_LSTM_SpatialViT_NoIntSup" \
#     --model-type-list "RawToTile_MobileNet" "TileToTile_LSTM" "TileToTileImage_SpatialViT" \
#     --omit-list "omit_no_xml" "omit_no_contour" \
#     --error-as-eval-loss \
#     --use-image-preds \
#     --batch-size 4 \
#     --series-length 2 \
#     --accumulate-grad-batches 8 \
#     --num-workers 6 \
#     --train-split-path './data/final_split/train_images_final.txt' \
#     --val-split-path './data/final_split/val_images_final.txt' \
#     --test-split-path './data/final_split/test_images_final.txt' \
#     --no-intermediate-supervision \
#     --no-early-stopping \
    
# python3.9 src/main.py \
#     --experiment-name "Final_MobileNet_LSTM_SpatialViT_NoOverlap" \
#     --model-type-list "RawToTile_MobileNet" "TileToTile_LSTM" "TileToTileImage_SpatialViT" \
#     --omit-list "omit_no_xml" "omit_no_contour" \
#     --error-as-eval-loss \
#     --use-image-preds \
#     --batch-size 4 \
#     --series-length 2 \
#     --accumulate-grad-batches 8 \
#     --num-workers 6 \
#     --train-split-path './data/final_split/train_images_final.txt' \
#     --val-split-path './data/final_split/val_images_final.txt' \
#     --test-split-path './data/final_split/test_images_final.txt' \
#     --resize-height 1344 \
#     --resize-width 1792 \
#     --crop-height 1120 \
#     --tile-overlap 0 \
#     --no-early-stopping \
    
# python3.9 src/main.py \
#     --experiment-name "Final_MobileNet_LSTM_SpatialViT_NoPreTile" \
#     --model-type-list "RawToTile_MobileNet_NoPreTile" "TileToTile_LSTM" "TileToTileImage_SpatialViT" \
#     --omit-list "omit_no_xml" "omit_no_contour" \
#     --error-as-eval-loss \
#     --use-image-preds \
#     --batch-size 4 \
#     --series-length 2 \
#     --accumulate-grad-batches 8 \
#     --num-workers 6 \
#     --train-split-path './data/final_split/train_images_final.txt' \
#     --val-split-path './data/final_split/val_images_final.txt' \
#     --test-split-path './data/final_split/test_images_final.txt' \
#     --resize-height 1344 \
#     --resize-width 1792 \
#     --crop-height 1120 \
#     --tile-overlap 0 \
#     --no-pre-tile \
#     --no-early-stopping \
    
# python3.9 src/main.py \
#     --experiment-name "Final_MobileNet_LSTM_SpatialViT_NoAugment" \
#     --model-type-list "RawToTile_MobileNet" "TileToTile_LSTM" "TileToTileImage_SpatialViT" \
#     --omit-list "omit_no_xml" "omit_no_contour" \
#     --error-as-eval-loss \
#     --use-image-preds \
#     --batch-size 4 \
#     --series-length 2 \
#     --accumulate-grad-batches 8 \
#     --num-workers 6 \
#     --train-split-path './data/final_split/train_images_final.txt' \
#     --val-split-path './data/final_split/val_images_final.txt' \
#     --test-split-path './data/final_split/test_images_final.txt' \
#     --no-flip-augment \
#     --no-resize-crop-augment \
#     --no-blur-augment \
#     --no-color-augment \
#     --no-brightness-contrast-augment \
#     --no-early-stopping \
    
# python3.9 src/main.py \
#     --experiment-name "Final_MobileNet_LSTM_SpatialViT_NoPretrain" \
#     --model-type-list "RawToTile_MobileNet" "TileToTile_LSTM" "TileToTileImage_SpatialViT" \
#     --omit-list "omit_no_xml" "omit_no_contour" \
#     --error-as-eval-loss \
#     --use-image-preds \
#     --batch-size 4 \
#     --series-length 2 \
#     --accumulate-grad-batches 8 \
#     --num-workers 6 \
#     --train-split-path './data/final_split/train_images_final.txt' \
#     --val-split-path './data/final_split/val_images_final.txt' \
#     --test-split-path './data/final_split/test_images_final.txt' \
#     --no-pretrain-backbone \
#     --no-early-stopping \
#     --max-epochs 15 \
#     --no-early-stopping \
    
# python3.9 src/main.py \
#     --experiment-name "Final_MobileNet_LSTM_SpatialViT_NoCropHeight" \
#     --model-type-list "RawToTile_MobileNet" "TileToTile_LSTM" "TileToTileImage_SpatialViT" \
#     --omit-list "omit_no_xml" "omit_no_contour" \
#     --error-as-eval-loss \
#     --use-image-preds \
#     --batch-size 2 \
#     --series-length 2 \
#     --accumulate-grad-batches 16 \
#     --num-workers 6 \
#     --train-split-path './data/final_split/train_images_final.txt' \
#     --val-split-path './data/final_split/val_images_final.txt' \
#     --test-split-path './data/final_split/test_images_final.txt' \
#     --crop-height 1244 \
#     --no-resize-crop-augment \
#     --no-early-stopping \
    
# python3.9 src/main.py \
#     --experiment-name "Final_MobileNet_LSTM_SpatialViT_NoImagePreds" \
#     --model-type-list "RawToTile_MobileNet" "TileToTile_LSTM" "TileToTileImage_SpatialViT" \
#     --omit-list "omit_no_xml" "omit_no_contour" \
#     --error-as-eval-loss \
#     --batch-size 4 \
#     --series-length 2 \
#     --accumulate-grad-batches 8 \
#     --num-workers 6 \
#     --train-split-path './data/final_split/train_images_final.txt' \
#     --val-split-path './data/final_split/val_images_final.txt' \
#     --test-split-path './data/final_split/test_images_final.txt' \
#     --no-early-stopping \
    
# python3.9 src/main.py \
#     --experiment-name "Final_MobileNet_LSTM_SpatialViT_100Resize" \
#     --model-type-list "RawToTile_MobileNet" "TileToTile_LSTM" "TileToTileImage_SpatialViT" \
#     --omit-list "omit_no_xml" "omit_no_contour" \
#     --error-as-eval-loss \
#     --use-image-preds \
#     --batch-size 2 \
#     --series-length 2 \
#     --accumulate-grad-batches 16 \
#     --num-workers 6 \
#     --train-split-path './data/final_split/train_images_final.txt' \
#     --val-split-path './data/final_split/val_images_final.txt' \
#     --test-split-path './data/final_split/test_images_final.txt' \
#     --resize-height 1546 \
#     --resize-width 2060 \
#     --crop-height 1244 \
#     --no-early-stopping \
    
python3.9 src/main.py \
    --experiment-name "Final_MobileNet_LSTM_SpatialViT_Series3" \
    --model-type-list "RawToTile_MobileNet" "TileToTile_LSTM" "TileToTileImage_SpatialViT" \
    --omit-list "omit_no_xml" "omit_no_contour" \
    --error-as-eval-loss \
    --use-image-preds \
    --batch-size 2 \
    --series-length 3 \
    --accumulate-grad-batches 16 \
    --num-workers 6 \
    --train-split-path './data/final_split/train_images_final.txt' \
    --val-split-path './data/final_split/val_images_final.txt' \
    --test-split-path './data/final_split/test_images_final.txt' \
    --no-early-stopping \
    
python3.9 src/main.py \
    --experiment-name "Final_MobileNet_LSTM_SpatialViT_Series4" \
    --model-type-list "RawToTile_MobileNet" "TileToTile_LSTM" "TileToTileImage_SpatialViT" \
    --omit-list "omit_no_xml" "omit_no_contour" \
    --error-as-eval-loss \
    --use-image-preds \
    --batch-size 2 \
    --series-length 4 \
    --accumulate-grad-batches 16 \
    --num-workers 6 \
    --train-split-path './data/final_split/train_images_final.txt' \
    --val-split-path './data/final_split/val_images_final.txt' \
    --test-split-path './data/final_split/test_images_final.txt' \
    --no-early-stopping \
    

    
