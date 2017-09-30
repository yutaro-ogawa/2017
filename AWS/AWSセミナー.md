# AWS
- 2017年9月30日 
- Amazonのクラウドサービスです
- 動画 アマゾンウェブサービスとは(youtube)がわかりやすい

## AWSの利点
- ２−３分でサーバーが立ち上げられる
- 導入コストが安い
- 学習コストが低い
- 日本語のメニューやヘルプデスク
- サービス機能を拡大しやすい

# AWSの認証
- 電話でやるときは静かな声でやるほうが良い

## LAMP環境構築
- Linux, Apache,  MySQL、PHP

### やること
- EC2
- SSH
- LAMPインストール
- PHP

1. コンソールでリージョンを東京に設定
2. EC2を選択
Amazon Linux AMI 2017.03.1 (HVM), SSD Volume Type

https://ap-northeast-1.console.aws.amazon.com/ec2/v2/home?region=ap-northeast-1#
3. ステップ 1: Amazon マシンイメージ (AMI) 
4. 


キーペア
鍵ファイル まあとりあえずtestで
 新しいキーペアを選択

インスタンスの作成


SSH
- windowsの方
鍵ファイルを変換する必要がある
puttyを使用する
http://docs.aws.amazon.com/ja_jp/AWSEC2/latest/UserGuide/putty.html
※Windowsは面倒そう・・・

- mac, linuxのかた
新しい鍵を作る。
インスタンスで接続をクリックで、コマンドが出てくる

ダッシュボードで、「接続」を押して、
鍵のあるディレクトリで、

chmod 400 test.pem
ssh -i "test.pem" ec2-user@ec2-13-114-141-100.ap-northeast-1.compute.amazonaws.com


EC2ってアスキーアートが出れば良い。

LAMPインストール

sudo yum install -y httpd24 php56 mysql55-server php56-ysqlnd

※いまEC２に入っているからyumで良い

Appachをインストール
sudo service httpd start
sudo chkconfig httpd on
sudo chkconfig --list httpd

sudo chkconfig --list httpd


ダッシュボードの左メニューから、セキュリティグループ
ファイアウォールを開ける
インバウンド　編集

選んで、編集で、追加できますよ、追加、HTTP　マイIP,
HTTP, HTTPS


パブリックDNSを打ち込むとページが出る
ec2-13-114-141-100.ap-northeast-1.compute.amazonaws.com

ファイルの追加削除および変更の許可を設定

sudo groupadd www
sudo usermod -a -G www ec2-user
exit

ssh -i "test.pem" ec2-user@ec2-13-114-141-100.ap-northeast-1.compute.amazonaws.com


groups

sudo chown -R root:www /var/www

sudo chmod 2775 /var/www

find /var/www -type d -exec sudo chmod 2775 {} \;
sudo find /var/www -type f -exec sudo chmod 0664 {} \;
echo "<?php echo 'Hello World'; ?>" > /var/www/html/index.php

パブリックDNSを打ち込むとページが出る
ec2-13-114-141-100.ap-northeast-1.compute.amazonaws.com


MySQL・phpMyAdmin


s

# 全体的所感
- 受講者に合わせてくれるが、進行はまずまず早い。
- コマンドラインとか少しは触ったことがあるくらいが良い


AIが使えるエンジニアでないとつらい
文系エンジニアは消えていく

最近のトレンド、話題

人間の脳初めてインターネットに接続

braitermet

脳とコンピュータをつなぐ
https://jp.sputniknews.com/science/201709184095984/


JAWS-UG
AWSのユーザー会　レベル高い
https://jaws-ug.jp/

