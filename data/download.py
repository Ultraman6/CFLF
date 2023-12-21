# import tensorflow_federated as tff
# import tensorflow
# only_digit = False
#
#
# def download_and_save_federated_emnist():
#     tff.simulation.datasets.emnist.load_data(cache_dir="./", only_digits=only_digit)
#
# # def download_femnist(dataset_root, train_file='fed_emnist_train.h5', test_file='fed_emnist_test.h5'):
# #     femnist_url = "YOUR_FEMNIST_DOWNLOAD_URL"  # 替换为FEMNIST数据集的下载链接
# #     femnist_zip = os.path.join(dataset_root, 'femnist.zip')
# #
# #     if not os.path.exists(femnist_zip):
# #         print(f"Downloading FEMNIST dataset to {femnist_zip}...")
# #         response = requests.get(femnist_url)
# #         with open(femnist_zip, 'wb') as f:
# #             f.write(response.content)
# #
# #         with zipfile.ZipFile(femnist_zip, 'r') as zip_ref:
# #             print(f"Extracting FEMNIST dataset to {dataset_root}...")
# #             zip_ref.extractall(dataset_root)
# #
# #     train_path = os.path.join(dataset_root, train_file)
# #     test_path = os.path.join(dataset_root, test_file)
# #
# #     if not os.path.exists(train_path) or not os.path.exists(test_path):
# #         raise RuntimeError("Failed to download or extract FEMNIST files.")
