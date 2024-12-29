import os
import re
import json
import pickle
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class ContextualMentalHealthChatbot:
    def __init__(self, json_path, context_path='models/context_data.json', max_words=5000, max_len=50, model_dir='models'):
        """
        Inisialisasi Chatbot dengan Parameter Kontekstual
        
        Args:
            json_path (str): Path file JSON training data
            context_path (str): Path file untuk menyimpan konteks percakapan
            max_words (int): Jumlah maksimum kata dalam tokenizer
            max_len (int): Panjang maksimum sekuens
            model_dir (str): Direktori untuk menyimpan model
        """
        # Konfigurasi
        self.json_path = json_path
        self.context_path = context_path
        self.max_words = max_words
        self.max_len = max_len
        self.model_dir = model_dir
        
        # Komponen model
        self.tokenizer = None
        self.label_encoder = None
        self.model = None
        self.training_data = []
        self.feedback_data = []
        
        # Manajemen konteks
        self.conversation_context = {
            'current_topic': None,
            'last_input': None,
            'context_chain': []
        }
        
        # Buat direktori model jika belum ada
        os.makedirs(model_dir, exist_ok=True)
        
        # Path file
        self.model_path = os.path.join(model_dir, 'contextual_chatbot_model.h5')
        self.vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
        self.intents_path = os.path.join(model_dir, 'intents.pkl')
        self.feedback_path = os.path.join(model_dir, 'feedback_data.json')
        
        # Load training data
        self.load_training_data()
        self.load_feedback_data()
        self.load_context_data()
    
    def load_context_data(self):
        """
        Memuat data konteks percakapan
        """
        try:
            if os.path.exists(self.context_path):
                with open(self.context_path, 'r', encoding='utf-8') as file:
                    context_data = json.load(file)
                    self.conversation_context = context_data
            else:
                self.conversation_context = {
                    'current_topic': None,
                    'last_input': None,
                    'context_chain': []
                }
        except Exception as e:
            print(f"Kesalahan memuat konteks: {e}")
            self.conversation_context = {
                'current_topic': None,
                'last_input': None,
                'context_chain': []
            }
    
    def save_context_data(self):
        """
        Menyimpan data konteks percakapan
        """
        try:
            with open(self.context_path, 'w', encoding='utf-8') as file:
                json.dump(self.conversation_context, file, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Kesalahan menyimpan konteks: {e}")
            
    def preprocess_text(self, text):
        """
        Preprocessing teks untuk tokenisasi
        
        Args:
            text (str): Teks input
        
        Returns:
            str: Teks yang sudah diproses
        """
        # Konversi ke huruf kecil
        text = str(text).lower()
        
        # Hapus tanda baca dan karakter khusus
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Hapus spasi berlebih
        text = ' '.join(text.split())
        
        return text

    def load_training_data(self):
        """
        Memuat dan memproses data latih dari file JSON
        """
        try:
            with open(self.json_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
                
                # Pastikan mengambil data dari kunci "data"
                self.training_data = json_data.get('data', [])
                
                # Preprocessing pola
                for intent in self.training_data:
                    intent['patterns'] = [
                        self.preprocess_text(pattern) 
                        for pattern in intent['patterns']
                    ]
        
        except Exception as e:
            print(f"Kesalahan memuat data: {e}")
            self.training_data = []

    def load_feedback_data(self):
        """
        Memuat data feedback yang tersimpan
        """
        try:
            if os.path.exists(self.feedback_path):
                with open(self.feedback_path, 'r', encoding='utf-8') as file:
                    self.feedback_data = json.load(file)
            else:
                self.feedback_data = []
        except Exception as e:
            print(f"Kesalahan memuat feedback: {e}")
            self.feedback_data = []

    def save_feedback_data(self):
        """
        Menyimpan data feedback
        """
        try:
            with open(self.feedback_path, 'w', encoding='utf-8') as file:
                json.dump(self.feedback_data, file, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Kesalahan menyimpan feedback: {e}")

    def add_feedback(self, user_input, user_response, bot_response):
        """
        Menambahkan data feedback untuk pembelajaran adaptif
        
        Args:
            user_input (str): Input pengguna
            user_response (str): Respons yang diinginkan pengguna
            bot_response (str): Respons bot sebelumnya
        """
        # Preprocessing input
        processed_input = self.preprocess_text(user_input)
        processed_response = self.preprocess_text(user_response)
        
        # Tambahkan ke data feedback
        feedback_entry = {
            'input': processed_input,
            'expected_response': processed_response,
            'bot_response': bot_response,
            'timestamp': tf.timestamp().numpy()
        }
        
        self.feedback_data.append(feedback_entry)
        self.save_feedback_data()
        
        print("Terima kasih atas masukan Anda. Saya akan belajar dari interaksi ini.")

    def train(self):
        """
        Melatih model chatbot dengan TF-IDF dan cosine similarity
        """
        try:
            # Gabungkan data latih asli dan feedback
            all_patterns = []
            for intent in self.training_data:
                all_patterns.extend(intent['patterns'])
            
            # Tambahkan input dari feedback
            for feedback in self.feedback_data:
                all_patterns.append(feedback['input'])
                all_patterns.append(feedback['expected_response'])
            
            # Buat TF-IDF Vectorizer
            self.vectorizer = TfidfVectorizer()
            pattern_vectors = self.vectorizer.fit_transform(all_patterns)
            
            # Simpan vectorizer
            with open(self.vectorizer_path, 'wb') as handle:
                pickle.dump(self.vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Simpan intent data
            with open(self.intents_path, 'wb') as handle:
                pickle.dump(self.training_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            print("Pelatihan model selesai dengan menggabungkan data feedback.")
        
        except Exception as e:
            print(f"Terjadi kesalahan saat melatih model: {e}")
            import traceback
            traceback.print_exc()

    def find_best_match(self, user_input, threshold=0.5):
        """
        Mencari intent terbaik berdasarkan kesamaan cosine
        
        Args:
            user_input (str): Input pengguna
            threshold (float): Ambang batas kesamaan
        
        Returns:
            dict: Intent terbaik atau None
        """
        # Preprocessing input
        processed_input = self.preprocess_text(user_input)
        
        # Muat vectorizer jika belum dimuat
        if not hasattr(self, 'vectorizer'):
            try:
                with open(self.vectorizer_path, 'rb') as handle:
                    self.vectorizer = pickle.load(handle)
                
                with open(self.intents_path, 'rb') as handle:
                    self.training_data = pickle.load(handle)
            except Exception as e:
                print(f"Kesalahan memuat model: {e}")
                return None
        
        # Vektorisasi input pengguna
        input_vector = self.vectorizer.transform([processed_input])
        
        # Simpan skor kesamaan untuk setiap intent
        best_match = None
        best_score = 0
        
        # Loop melalui setiap intent
        for intent in self.training_data:
            # Vektorisasi semua pola intent
            intent_patterns = intent['patterns']
            intent_vectors = self.vectorizer.transform(intent_patterns)
            
            # Hitung cosine similarity
            similarities = cosine_similarity(input_vector, intent_vectors)
            max_similarity = np.max(similarities)
            
            # Perbarui match terbaik
            if max_similarity > best_score:
                best_score = max_similarity
                best_match = intent
        
        # Periksa feedback data
        for feedback in self.feedback_data:
            feedback_vector = self.vectorizer.transform([feedback['input']])
            similarity = cosine_similarity(input_vector, feedback_vector)[0][0]
            
            if similarity > best_score:
                best_score = similarity
                best_match = {
                    'tag': 'feedback',
                    'patterns': [feedback['input']],
                    'responses': [feedback['expected_response']]
                }
        
        # Kembalikan intent jika melebihi ambang batas
        if best_score >= threshold:
            return {
                'intent': best_match,
                'confidence': best_score
            }
        
        return None
    
    def update_context(self, input_text, intent_tag):
        """
        Memperbarui konteks percakapan
        
        Args:
            input_text (str): Input pengguna
            intent_tag (str): Tag intent yang terdeteksi
        """
        # Tambahkan input ke rantai konteks
        self.conversation_context['context_chain'].append({
            'input': input_text,
            'tag': intent_tag
        })
        
        # Batasi panjang rantai konteks
        if len(self.conversation_context['context_chain']) > 5:
            self.conversation_context['context_chain'].pop(0)
        
        # Perbarui topik dan input terakhir
        self.conversation_context['current_topic'] = intent_tag
        self.conversation_context['last_input'] = input_text
        
        # Simpan konteks
        self.save_context_data()
    
    def find_contextual_response(self, current_input):
        """
        Mencari respons berdasarkan konteks percakapan
        
        Args:
            current_input (str): Input pengguna saat ini
        
        Returns:
            dict: Respons kontekstual atau None
        """
        # Daftar topik khusus dengan respons berurutan
        contextual_topics = {
            'tidur': [
                {
                    'patterns': ['sulit tidur', 'tidak bisa tidur', 'insomnia'],
                    'follow_up': [
                        "Apakah Anda sudah mencoba menerapkan hygiene tidur?",
                        "Baiklah, hygiene tidur adalah praktik yang membantu meningkatkan kualitas tidur. Beberapa tips meliputi:",
                        "Pertama, atur jadwal tidur yang konsisten. Tidur dan bangun pada waktu yang sama setiap hari.",
                        "Hindari penggunaan elektronik sebelum tidur karena cahaya biru dapat mengganggu produksi melatonin.",
                        "Ciptakan lingkungan tidur yang nyaman: gelap, tenang, dan sejuk.",
                        "Hindari konsumsi kafein, alkohol, dan makanan berat menjelang tidur.",
                        "Lakukan relaksasi sebelum tidur seperti meditasi atau membaca buku."
                    ]
                },
                {
                    'patterns': ['belum'],
                    'follow_up': [
                        "Mari kita bahas beberapa strategi untuk meningkatkan kualitas tidur Anda.",
                        "Pertama, cobalah membuat rutinitas tidur yang teratur.",
                        "Gunakan teknik relaksasi seperti pernapasan dalam atau meditasi sebelum tidur.",
                        "Hindari penggunaan ponsel atau komputer minimal 1 jam sebelum tidur.",
                        "Jika masih sulit, pertimbangkan untuk berkonsultasi dengan profesional kesehatan."
                    ]
                }
            ]
        }
        
        # Preprocessing input
        processed_input = self.preprocess_text(current_input)
        
        # Cek konteks sebelumnya
        if self.conversation_context['current_topic']:
            topic = self.conversation_context['current_topic']
            last_input = self.conversation_context['last_input']
            
            # Cari topik dalam daftar topik khusus
            if topic in contextual_topics:
                for context_item in contextual_topics[topic]:
                    # Cocokkan pola dengan input terakhir
                    for pattern in context_item.get('patterns', []):
                        if pattern.lower() in last_input.lower():
                            # Pilih respons berurutan
                            if 'follow_up' in context_item:
                                return {
                                    'tag': topic,
                                    'response': context_item['follow_up'][0],
                                    'context_responses': context_item['follow_up'][1:],
                                    'is_contextual': True
                                }
        
        return None
    
    def get_response(self, user_input):
        """
        Mendapatkan respons terbaik untuk input pengguna
        """
        # Cek respons kontekstual terlebih dahulu
        contextual_response = self.find_contextual_response(user_input)
        if contextual_response:
            # Update konteks dengan respons kontekstual
            self.update_context(user_input, contextual_response['tag'])
            return {
                'tag': contextual_response['tag'],
                'confidence': 1.0,
                'response': contextual_response['response'],
                'matched_pattern': user_input,
                'is_contextual': True,
                'context_responses': contextual_response.get('context_responses', [])
            }
        
        # Jika tidak ada respons kontekstual, gunakan metode pencarian intent
        # Cari intent terdekat
        match = self.find_best_match(user_input)
        
        if match:
            # Pilih respons secara acak dari intent yang cocok
            if match['intent']['tag'] == 'feedback':
                response = match['intent']['responses'][0]
            else:
                response = random.choice(match['intent']['responses'])
            
            # Update konteks
            self.update_context(user_input, match['intent']['tag'])
            
            return {
                'tag': match['intent']['tag'],
                'confidence': float(match['confidence']),
                'response': response,
                'matched_pattern': user_input,
                'is_contextual': False
            }
        
        # Respons default jika tidak ada kecocokan
        return {
            'tag': 'unknown',
            'confidence': 0.0,
            'response': "Maaf, saya tidak mengerti pertanyaan Anda. Bisa diulangi? Atau Anda bisa memberi tahu saya jawaban yang seharusnya.",
            'matched_pattern': user_input,
            'is_contextual': False
        }

    def interactive_chat(self):
        """
        Memulai sesi chat interaktif
        """
        print("Chatbot: Hai! Saya siap membantu Anda.")
        while True:
            user_input = input("Anda: ")
            
            # Keluar dari percakapan
            if user_input.lower() in ['keluar', 'exit', 'bye']:
                print("Chatbot: Terima kasih telah berbicara denganku. Semoga harimu menyenangkan!")
                break
            
            # Cari intent untuk input pengguna
            result = self.get_response(user_input)
            
            # Periksa tag untuk mengakhiri percakapan
            if result['tag'] == 'selesai':
                print(f"Chatbot: {result['response']}")
                break
            
            # Tampilkan respons
            print(f"Chatbot: {result['response']}")
            
            # Tampilkan informasi tambahan jika perlu
            if result.get('is_contextual', False):
                contextual_responses = result.get('context_responses', [])
                for resp in contextual_responses:
                    lanjut = input("Lanjutkan? (ya/tidak): ").lower()
                    if lanjut == 'ya':
                        print(f"Chatbot: {resp}")
                    else:
                        break
            
            # Cetak detail kategori dan confidence
            print(f"(Kategori: {result['tag']}, Confidence: {result['confidence']:.2f})")
            
            # Jika tidak yakin, minta masukan pengguna
            if result['tag'] == 'unknown':
                lanjut = input("Apakah Anda ingin memberi tahu saya jawaban yang benar? (ya/tidak): ").lower()
                if lanjut == 'ya':
                    user_response = input("Silakan berikan jawaban yang seharusnya: ")
                    self.add_feedback(user_input, user_response, result['response'])
                    
                    # Retrain model dengan data baru
                    self.train()


class AdaptiveMentalHealthChatbot:
    def __init__(self, json_path, max_words=5000, max_len=50, model_dir='models'):
        """
        Inisialisasi Chatbot dengan Parameter Konfigurasi
        
        Args:
            json_path (str): Path file JSON training data
            max_words (int): Jumlah maksimum kata dalam tokenizer
            max_len (int): Panjang maksimum sekuens
            model_dir (str): Direktori untuk menyimpan model
        """
        # Konfigurasi
        self.json_path = json_path
        self.max_words = max_words
        self.max_len = max_len
        self.model_dir = model_dir
        
        # Komponen model
        self.tokenizer = None
        self.label_encoder = None
        self.model = None
        self.training_data = []
        self.feedback_data = []
        
        # Buat direktori model jika belum ada
        os.makedirs(model_dir, exist_ok=True)
        
        # Path file
        self.model_path = os.path.join(model_dir, 'adaptive_chatbot_model.h5')
        self.vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
        self.intents_path = os.path.join(model_dir, 'intents.pkl')
        self.feedback_path = os.path.join(model_dir, 'feedback_data.json')
        
        # Load training data
        self.load_training_data()
        self.load_feedback_data()
    
    def preprocess_text(self, text):
        """
        Preprocessing teks untuk tokenisasi
        
        Args:
            text (str): Teks input
        
        Returns:
            str: Teks yang sudah diproses
        """
        # Konversi ke huruf kecil
        text = str(text).lower()
        
        # Hapus tanda baca dan karakter khusus
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Hapus spasi berlebih
        text = ' '.join(text.split())
        
        return text
    
    def load_training_data(self):
        """
        Memuat dan memproses data latih dari file JSON
        """
        try:
            with open(self.json_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
                
                # Pastikan mengambil data dari kunci "data"
                self.training_data = json_data.get('data', [])
                
                # Preprocessing pola
                for intent in self.training_data:
                    intent['patterns'] = [
                        self.preprocess_text(pattern) 
                        for pattern in intent['patterns']
                    ]
        
        except Exception as e:
            print(f"Kesalahan memuat data: {e}")
            self.training_data = []
    
    def load_feedback_data(self):
        """
        Memuat data feedback yang tersimpan
        """
        try:
            if os.path.exists(self.feedback_path):
                with open(self.feedback_path, 'r', encoding='utf-8') as file:
                    self.feedback_data = json.load(file)
            else:
                self.feedback_data = []
        except Exception as e:
            print(f"Kesalahan memuat feedback: {e}")
            self.feedback_data = []
    
    def save_feedback_data(self):
        """
        Menyimpan data feedback
        """
        try:
            with open(self.feedback_path, 'w', encoding='utf-8') as file:
                json.dump(self.feedback_data, file, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Kesalahan menyimpan feedback: {e}")
    
    def add_feedback(self, user_input, user_response, bot_response):
        """
        Menambahkan data feedback untuk pembelajaran adaptif
        
        Args:
            user_input (str): Input pengguna
            user_response (str): Respons yang diinginkan pengguna
            bot_response (str): Respons bot sebelumnya
        """
        # Preprocessing input
        processed_input = self.preprocess_text(user_input)
        processed_response = self.preprocess_text(user_response)
        
        # Tambahkan ke data feedback
        feedback_entry = {
            'input': processed_input,
            'expected_response': processed_response,
            'bot_response': bot_response,
            'timestamp': tf.timestamp().numpy()
        }
        
        self.feedback_data.append(feedback_entry)
        self.save_feedback_data()
        
        print("Terima kasih atas masukan Anda. Saya akan belajar dari interaksi ini.")
    
    def train(self):
        """
        Melatih model chatbot dengan TF-IDF dan cosine similarity
        """
        try:
            # Gabungkan data latih asli dan feedback
            all_patterns = []
            for intent in self.training_data:
                all_patterns.extend(intent['patterns'])
            
            # Tambahkan input dari feedback
            for feedback in self.feedback_data:
                all_patterns.append(feedback['input'])
                all_patterns.append(feedback['expected_response'])
            
            # Buat TF-IDF Vectorizer
            self.vectorizer = TfidfVectorizer()
            pattern_vectors = self.vectorizer.fit_transform(all_patterns)
            
            # Simpan vectorizer
            with open(self.vectorizer_path, 'wb') as handle:
                pickle.dump(self.vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Simpan intent data
            with open(self.intents_path, 'wb') as handle:
                pickle.dump(self.training_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            print("Pelatihan model selesai dengan menggabungkan data feedback.")
        
        except Exception as e:
            print(f"Terjadi kesalahan saat melatih model: {e}")
            import traceback
            traceback.print_exc()
    
    def find_best_match(self, user_input, threshold=0.5):
        """
        Mencari intent terbaik berdasarkan kesamaan cosine
        
        Args:
            user_input (str): Input pengguna
            threshold (float): Ambang batas kesamaan
        
        Returns:
            dict: Intent terbaik atau None
        """
        # Preprocessing input
        processed_input = self.preprocess_text(user_input)
        
        # Muat vectorizer jika belum dimuat
        if not hasattr(self, 'vectorizer'):
            try:
                with open(self.vectorizer_path, 'rb') as handle:
                    self.vectorizer = pickle.load(handle)
                
                with open(self.intents_path, 'rb') as handle:
                    self.training_data = pickle.load(handle)
            except Exception as e:
                print(f"Kesalahan memuat model: {e}")
                return None
        
        # Vektorisasi input pengguna
        input_vector = self.vectorizer.transform([processed_input])
        
        # Simpan skor kesamaan untuk setiap intent
        best_match = None
        best_score = 0
        
        # Loop melalui setiap intent
        for intent in self.training_data:
            # Vektorisasi semua pola intent
            intent_patterns = intent['patterns']
            intent_vectors = self.vectorizer.transform(intent_patterns)
            
            # Hitung cosine similarity
            similarities = cosine_similarity(input_vector, intent_vectors)
            max_similarity = np.max(similarities)
            
            # Perbarui match terbaik
            if max_similarity > best_score:
                best_score = max_similarity
                best_match = intent
        
        # Periksa feedback data
        for feedback in self.feedback_data:
            feedback_vector = self.vectorizer.transform([feedback['input']])
            similarity = cosine_similarity(input_vector, feedback_vector)[0][0]
            
            if similarity > best_score:
                best_score = similarity
                best_match = {
                    'tag': 'feedback',
                    'patterns': [feedback['input']],
                    'responses': [feedback['expected_response']]
                }
        
        # Kembalikan intent jika melebihi ambang batas
        if best_score >= threshold:
            return {
                'intent': best_match,
                'confidence': best_score
            }
        
        return None
    
    def get_response(self, user_input):
        """
        Mendapatkan respons terbaik untuk input pengguna
        """
        # Cek respons kontekstual terlebih dahulu
        contextual_response = self.find_contextual_response(user_input)
        if contextual_response:
            # Update konteks dengan respons kontekstual
            self.update_context(user_input, contextual_response['tag'])
            return {
                'tag': contextual_response['tag'],
                'confidence': 1.0,
                'response': contextual_response['response'],
                'matched_pattern': user_input,
                'is_contextual': True
            }
        
        # Jika tidak ada respons kontekstual, gunakan metode sebelumnya
        # Cari intent terdekat
        match = self.find_best_match(user_input)
        
        if match:
            # Pilih respons secara acak dari intent yang cocok
            if match['intent']['tag'] == 'feedback':
                response = match['intent']['responses'][0]
            else:
                response = random.choice(match['intent']['responses'])
            
            # Update konteks
            self.update_context(user_input, match['intent']['tag'])
            
            return {
                'tag': match['intent']['tag'],
                'confidence': float(match['confidence']),
                'response': response,
                'matched_pattern': user_input,
                'is_contextual': False
            }
        
        # Respons default jika tidak ada kecocokan
        return {
            'tag': 'unknown',
            'confidence': 0.0,
            'response': "Maaf, saya tidak mengerti pertanyaan Anda. Bisa diulangi? Atau Anda bisa memberi tahu saya jawaban yang seharusnya.",
            'matched_pattern': user_input,
            'is_contextual': False
        }
    
    def interactive_chat(self):
        """
        Memulai sesi chat interaktif
        """
        print("Chatbot: Hai! Saya siap membantu Anda.")
        while True:
            user_input = input("Anda: ")
            
            # Cari intent untuk input pengguna
            result = self.get_response(user_input)
            
            # Periksa tag untuk mengakhiri percakapan
            if result['tag'] == 'selesai':
                print(f"Chatbot: {result['response']}")
                break
            
            # Tampilkan respons
            print(f"Chatbot: {result['response']}")
            
            # Tampilkan informasi tambahan jika perlu
            if result.get('is_contextual', False):
                contextual_responses = result.get('context_responses', [])
                for resp in contextual_responses:
                    lanjut = input("Lanjutkan? (ya/tidak): ").lower()
                    if lanjut == 'ya':
                        print(f"Chatbot: {resp}")
                    else:
                        break
            
            # Cetak detail kategori dan confidence
            print(f"(Kategori: {result['tag']}, Confidence: {result['confidence']:.2f})")
            
            # Jika tidak yakin, minta masukan pengguna
            if result['tag'] == 'unknown':
                lanjut = input("Apakah Anda ingin memberi tahu saya jawaban yang benar? (ya/tidak): ").lower()
                if lanjut == 'ya':
                    user_response = input("Silakan berikan jawaban yang seharusnya: ")
                    self.add_feedback(user_input, user_response, result['response'])
                    
                    # Retrain model dengan data baru
                    self.train()

# Contoh penggunaan
if __name__ == "__main__":
    # Pilih mode
    mode = input("Pilih mode (train/chat): ").lower()
    
    # Inisialisasi chatbot
    chatbot = ContextualMentalHealthChatbot('data.json')
    
    if mode == 'train':
        # Latih model
        chatbot.train()
    elif mode == 'chat':
        # Mulai chat interaktif
        chatbot.interactive_chat()
    else:
        print("Mode tidak valid. Gunakan 'train' atau 'chat'.")
