# ディープラーニングでスリーサイズを予想

## セクシャリティに関して議論がある中で行ったということ
身体的特徴が、何らか数値化されるのは気分のいいものではありません。  
しかし、実生活上、服を購入する際に、メジャーでぐるっと測られるのは必須であり、私はこれがかなり苦手です。  
というか知らない人に手で触れるのが苦手で、このプロセスを省略したいと考えています。  
ディープラーニングでは内部的に和算と積算などを計算ができる様子が観測できます（RNNで足し算引き算掛け算割り算ができる）。  
こういう特性があることから、KPIの予想に使えることが想像できるかと思います。  
実際、ドワンゴの研究ではユーザの閲覧数を深層学習で予想するなどを行っておりうまく行っているようです[1]。  
数値化されて（もちろん、望まない限り行われないが原則）便利になる領域であれば、開拓する意義はあると考えています。  

※1. これは完全なる余暇を利用した完全なる個人の研究であり、所属する団体やなにやらを代表するものではありません、念のため  
※2. 余談ですが、男性の写真から年収を予想する回帰も余暇で行っており、女性ばかり、こういった評価の対象になることを避け、対称性を維持したいです  

## 前処理1. データを集める
身体的特徴を商業的に利用しているグラビア女優に関して、予想を行います。  

収集対象の女優一覧はこちらの[Wikipediaの一覧](https://ja.wikipedia.org/wiki/%E3%82%B0%E3%83%A9%E3%83%93%E3%82%A2%E3%82%A2%E3%82%A4%E3%83%89%E3%83%AB%E4%B8%80%E8%A6%A7)を用いました。

古すぎる写真だとインターネット上にデータが存在しなかったり、年代ごとの文化の違いからうまく学習できなかったりするので、1980年台以降に生まれた方に絞っています。  
収集方法は、Microsoft社のBingという検索エンジンに女優の名前をクエリに入力し、画像を集めます。スクレイパーは[自分で設計したもの](https://github.com/GINK03/kotlin-headlessbrowser-selenium-jsoup-parser)です。 

## 前処理2. スクレイピングで集めた必要でない写真（本人以外の写真など）を削る  

途中で、大量の検索対象の方と関係のないノイズデータが含まれてしまうのですが、ニートをつかったノイズフィルタリングシステムを導入することで、この問題をクリアしました[2]  
わざわざこのために、ネットワークを一個設計することになったので、結構労力必要です。

<p align="center">
  <img width="650px" src="https://cloud.githubusercontent.com/assets/4949982/25303776/abf6ec4e-2794-11e7-97a0-d0ff4df74e23.png">
</p>
<div align="center"> 図2. ノイズフィルタリングシステム </div>

## 活性化関数の検討
数値化されるデータはバスト80cm, ウェスト58cm, ヒップ82cmなどというデータです。  
ディープラーニングで扱うには大きすぎる値のため、0.0 ~ 1.0程度の値におさめている必要があります（しかし、別に超えてもいい）
そこで乱暴ですがすべての値を100.0で割ってしまうことにしました。
```sh
(B80cm, W52cm, H82cm) -> (0.80, 0.52, 0.82)
```
ロジットにいれるとか色々考えたのですが、100cmを超える方も存在するので、そうなると予想できなくなるので、リニアな出力に単純にmean squared errorを取りました。

検討していないアプローチですが、例えば150cm位を上限として、各パラメータをとってロジットをとってmean squared errorを取るというのもありかもしれません。

## ネットワークの選択
ResNetでいいでしょう。  
私物のGPUなので、そんなに火力はないので、ResNet50で転移学習させました。  
ResNetはResidual Networkの通り残差を学習していきます。精度や表現パターンを学習するなら、これにまさるものは今は無いように思えます。（存在するのなら知らないだけなので許してください）
<p align="center">
  <img width="400px" src="https://deepage.net/img/resnet/residual_block.jpg">
</p>
<div align="center"> 図3. Residual Networkの残差計算の様子(picsource deepage.net) </div>

## 学習＆評価
・学習用データセット 35000枚の写真  
・評価用データセット  4311枚の写真  

MSEを学習するだけのなので、過学習にならないように注意しながらやってみましょう。
Testデータにおいて、MSEの値が途中から上昇する現象が発生するはずです。そこから過学習が発生していと考えられるので、学習はそこで打ち切りです。  

## 予想してみる
学習用データセットに含まれていらっしゃらない方で水着でメディアに露出されている方がいます。  
去年、ドラマで有名になったガッキーや田中麗奈さん、上坂すみれさん、清水あいりさんを予想してみました。

<p align="center">
  <img width="500px" src="https://cloud.githubusercontent.com/assets/4949982/25573100/ff51181c-2e7c-11e7-8a25-35035a206e51.png">
</p>
<div align="center"> 図4. 稲垣ゆいさん </div>

<p align="center">
  <img width="500px" src="https://cloud.githubusercontent.com/assets/4949982/25573109/17a49f7e-2e7d-11e7-9416-fd4b9d4f9bc4.png">
</p>
<div align="center"> 図5. 田中麗奈さん </div>

<p align="center">
  <img width="500px" src="https://cloud.githubusercontent.com/assets/4949982/25573116/2cd0a2a8-2e7d-11e7-97b5-04cc4ff96731.png">
</p>
<div align="center"> 図6. 上坂すみれさん </div>

<p align="center">
  <img width="500px" src="https://cloud.githubusercontent.com/assets/4949982/25574056/9047c8ae-2e85-11e7-97ee-ddde8a304cda.png">
</p>
<div align="center"> 図7. 清水あいりさん </div>

## 独自データセットの学習
　どなたでも環境があれば学習できます。  
　コードはgithub上で公開しており、再配布は自由ですが商用利用はおやめください。  
```sh
$ git clone https://github.com/GINK03/keras-bwh-predictor
```
　まず、学習したい対象の画像を縮小します。  
  ResNetを使う場合は224x224のサイズなのでそのサイズにリサイズして、何もない画像に貼り付けます。  
<p align="center">
  <img widht="450px" src="https://cloud.githubusercontent.com/assets/4949982/25574490/dc7c252c-2e89-11e7-9a9e-f210a95adc66.png">
</p>
<div align="center"> 図8. 画像を縮小します。 </div>

 そして、特定のディレクトリに、決まった規則の名前で保存してください。（名前がキーとなります）
<p align="center">
  <img widht="450px" src="https://cloud.githubusercontent.com/assets/4949982/25574414/28d6f0d8-2e89-11e7-908c-5a87ee2b9753.png">
</p>
<div align="center"> 図9. 特定のキーに保存します。 </div>

　bwh.txtファイルを編集して予想したい三つのパラメータを記述します。
<p align="center">
  <img width="200px" src="https://cloud.githubusercontent.com/assets/4949982/25574599/db48698a-2e8a-11e7-86a4-4e9cce62cc81.png">
</p>
　コードなかに、trainMaxという変数があり、学習に使うデータの最大値を決定しているパラメータがあるので、適宜編集してください。
　 配置が完了したら、学習です。  
  
 ```sh
 $ python3 deep_bwh.py --train
 ```
 
 何回目のepochが良いか、--evalという引数で評価できます。(出力される値が少ないほどよい)
 ```sh
 $ python3 deep_bwh.py --eval
 ```
 
 任意の画像のbwhを予想します。予想する画像はto_predに入れておいてください
 ```sh
 $ python3 deep_bwh.py --pred
 ```

 
## 感想
　上坂すみれさんのような写真に対して反応するので、大きさはある程度わかっているのかなと言う印象があります。  
 　今後の改善点としてグラビアなど芸能界特有の盛る現象とかあると思うので、下着メーカや水着メーカが頑張ってきれいなデータセットを揃えてくれれば、実用の可能性はあるように思えます。  
  お店でメジャーで測るんじゃなくて、スマホで自撮りすると、自分のプライベートな値が管理できるようになって、ネットとかで通販ができるようになると良いですね。  

## 参考
[1] [ドワンゴ視聴数予想](http://www.itmedia.co.jp/news/articles/1511/04/news114.html)  
[2] [ディープ前処理ツールキット](https://bitbucket.org/nardtree/maeshori-toolkit-for-deeplearning)
