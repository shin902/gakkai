# 機械学習を利用した大気のゆらぎのゆらぎの補正技術の開発

## フォルダ構成
- ./src - ソースファイル。以上！
- ./Resources - 画像など

## ./srcフォルダについて
### Modules
- 自作モジュール（importで読み込むやつ）を格納するフォルダ
- 単体で実行した場合も、テスト、確認用のコードが実行される
---
- noise2noise.py
    - Noise2Noise(Noise_to_Noise)を実装したコード
    - U-Netを使用している
    - [U-Netとは | スキルアップAI Journal](https://www.skillupai.com/blog/tech/segmentation2/)
- ellipse.py
    - アフィン変換（回転、拡大縮小、平行移動）によって楕円補正を試みたやつ
- generate_movie.py
  - 画像の入ったフォルダから、動画を生成するコード
  - 星グル写真のタイムラプスをイメージするとわかりやすいかも
- hsv_low_high.ipynb
  - hsvによる木星の模様検出を試みた、jupyter notebook用のファイル
  - 普通に開くととてつもなく見にくいため、Google Colab上で、閲覧すること推奨
- hsv.py
  - 上のファイルをモジュール化したもの
  - なぜか両方とも結果が違う
- sift.py
  - 特徴点抽出、特徴点マッチングを利用した楕円補正
  - ホモグラフィー変換を使用するため、特徴点マッチングがうまくいかないと失敗する
  - hsv.pyと組み合わせることを想定
  - 簡単なノイズ除去をするクラスも同梱
- ML/affine_image_correction_r1.py
  - 強化学習による、楕円補正の実装（開発中）
  - 2つの画像のシャープネスを測定し、値がいいほうにもう片方を補正する
- ML/image_sharpness.py
  - 使用してないはず（AIに丸々実装させたからわからん）

## mainフォルダについて
- これらのモジュールを使用して、実際にコードを実装したコードを格納するフォルダ
---
- movie_denoise.py
  - Modules/noise2noise.pyを使用し、フォルダ内のすべての画像ファイルをノイズ除去する
  - ノイズ除去された連番画像から動画ファイルが生成される
- movie_affine.py
  - Modules/ellipse.pyを使用し、フォルダ内のすべての画像ファイルにアフィン変換を適用する
