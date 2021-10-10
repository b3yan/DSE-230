hadoop fs -mkdir /data
echo "Loading the datasets"
hadoop fs -copyFromLocal customers_dataset.csv /data/
hadoop fs -copyFromLocal customer_reviews_dataset.csv /data/
hadoop fs -copyFromLocal geolocation_dataset.csv /data/
hadoop fs -copyFromLocal orders_dataset.csv /data/
hadoop fs -copyFromLocal order_items_dataset.csv /data/
hadoop fs -copyFromLocal order_payments_dataset.csv /data/
hadoop fs -copyFromLocal sellers_dataset.csv /data/
hadoop fs -copyFromLocal products_dataset.csv /data/
hadoop fs -copyFromLocal product_category_name_translation.csv /data/
echo "Listing the loaded files"
hadoop fs -ls /data
