import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing
from tqdm import tqdm

VOCAB_SIZE = 32000  # Taille du vocabulaire cible pour le tokenizer BPE


def iter_dataset(data_dir):
    """Yield raw text content from all .txt files in data_dir."""
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.txt')]
    print(f"📂 {len(files):,} fichiers .txt trouvés dans {data_dir}/")

    for file_path in tqdm(files, desc="📖 Lecture"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"⚠️  Skip {file_path}: {e}")
            continue

        if content.strip():
            yield content


def train_bpe_tokenizer(data_dir='data', vocab_size=VOCAB_SIZE,
                        output_path='bpe_tokenizer_32k.json', force=False):

    if os.path.exists(output_path) and not force:
        print(f"⚠️  Tokenizer existant trouvé: {output_path}")
        print(f"   Supprimez le fichier ou passez force=True pour le ré-entraîner")
        return output_path

    print(f"\n{'='*60}")
    print(f"🔧 Entraînement du tokenizer BPE")
    print(f"{'='*60}")
    print(f"🔒 Vocabulaire cible: {vocab_size:,} tokens")
    print(f"📂 Données: {data_dir}/ (TOUS les fichiers, sans limite)")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    # ByteLevel pre-tokenizer (comme GPT-2/GPT-3/GPT-4)
    # Gère tous les caractères Unicode (emojis, accents, code, etc.)
    tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=True, use_regex=True)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        show_progress=True,
        min_frequency=2,
        initial_alphabet=ByteLevelPreTokenizer.alphabet(),
    )

    print(f"🔄 Entraînement en cours sur l'ensemble du dataset...")
    tokenizer.train_from_iterator(iter_dataset(data_dir), trainer=trainer)

    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> $B:1 <eos>:1",
        special_tokens=[("<bos>", 2), ("<eos>", 3)],
    )

    tokenizer.save(output_path)

    vocab = tokenizer.get_vocab()
    print(f"\n✅ Tokenizer BPE entraîné avec succès!")
    print(f"🔒 Taille du vocabulaire: {len(vocab):,} tokens")
    print(f"💾 Sauvegardé: {output_path}")

    # Verify special tokens
    for tok in ["<pad>", "<unk>", "<bos>", "<eos>"]:
        tid = tokenizer.token_to_id(tok)
        print(f"   {tok:8s} -> id {tid}")

    return output_path


def main():
    # Anchor to project root so data/ and the output tokenizer resolve from
    # anywhere (this script lives in scripts/).
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    data_dir = str(project_root / 'data')

    if not os.path.exists(data_dir):
        print(f"❌ Erreur: Le répertoire '{data_dir}' n'existe pas!")
        print(f"   Créez un répertoire avec vos fichiers .txt")
        return

    tokenizer_path = train_bpe_tokenizer(
        data_dir=data_dir,
        vocab_size=VOCAB_SIZE,
        output_path=str(project_root / 'bpe_tokenizer_32k.json'),
        force=False,
    )

    print("\n" + "=" * 60)
    print("🎉 Entraînement terminé!")
    print("=" * 60)
    print(f"📝 Tokenizer BPE: {tokenizer_path}")


if __name__ == "__main__":
    main()
