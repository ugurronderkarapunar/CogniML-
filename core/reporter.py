import openai

def generate_report(results, best_name, task, api_key=None):
    """Sonuçları metin rapora dönüştürür, opsiyonel LLM yorumu ekler."""
    lines = []
    lines.append("## 📊 CogniML Otomatik Rapor\n")
    lines.append(f"**Görev:** {task.upper()}")
    lines.append(f"**En Başarılı Model:** {best_name.upper()}\n")
    lines.append("### Model Performansları")
    for r in results:
        lines.append("- " + ", ".join(f"{k}: {v}" for k,v in r.items()))

    if api_key:
        openai.api_key = api_key
        prompt = f"Aşağıdaki makine öğrenmesi sonuçlarını yönetici özeti şeklinde yorumla. Hangi model daha iyi, neden? İş tavsiyesi ver.\n\n{results}"
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role":"user","content":prompt}],
                temperature=0.3
            )
            ai_comment = resp.choices[0].message.content
            lines.append("\n### 🤖 Yapay Zekâ Yorumu\n")
            lines.append(ai_comment)
        except Exception as e:
            lines.append(f"\n*(LLM yorumu alınamadı: {e})*")
    else:
        lines.append("\n*(LLM yorumu için OpenAI API anahtarı girin.)*")

    return "\n".join(lines)
