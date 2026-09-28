"""Comprehensive multilingual tests for ASHA-Sahaayak healthcare engine.

Tests cover all 12 supported languages:
- English (en), Hindi (hi), Marathi (mr), Telugu (te) - existing
- Bengali (bn), Tamil (ta), Gujarati (gu), Kannada (kn)
- Malayalam (ml), Odia (or), Punjabi (pa), Assamese (as)
"""

from pathlib import Path
import sys
import unittest


BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from languages import SUPPORTED_LANGUAGES, validate_language
from healthcare_engine import process_healthcare_input


class LanguageValidationTests(unittest.TestCase):
    def test_all_twelve_language_codes_are_accepted(self) -> None:
        for code in SUPPORTED_LANGUAGES:
            with self.subTest(code=code):
                self.assertEqual(validate_language(code), code)

    def test_case_insensitive_validation(self) -> None:
        self.assertEqual(validate_language("EN"), "en")
        self.assertEqual(validate_language("Hi"), "hi")
        self.assertEqual(validate_language("BN"), "bn")

    def test_invalid_code_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            validate_language("xx")
        with self.assertRaises(ValueError):
            validate_language("hindi")
        with self.assertRaises(ValueError):
            validate_language("")


class SymptomDetectionTests(unittest.TestCase):
    """Detect canonical symptoms from realistic native-script input for every language."""

    def _assert_symptom_detected(self, text: str, expected_symptom: str) -> None:
        result = process_healthcare_input(text)
        self.assertIn(expected_symptom, result["symptoms"],
                      f"Expected '{expected_symptom}' in {result['symptoms']} for input: {text}")

    def _assert_symptom_not_detected(self, text: str, symptom: str) -> None:
        result = process_healthcare_input(text)
        self.assertNotIn(symptom, result["symptoms"],
                         f"Did not expect '{symptom}' in {result['symptoms']} for input: {text}")

    # --- FEVER ---
    def test_fever_english(self):
        self._assert_symptom_detected("Patient has fever since 2 days", "fever")

    def test_fever_hindi(self):
        self._assert_symptom_detected("बुखार है 2 दिन से", "fever")

    def test_fever_marathi(self):
        self._assert_symptom_detected("ताप येत आहे 2 दिवसांपासून", "fever")

    def test_fever_telugu(self):
        self._assert_symptom_detected("జ్వరం ఉంది 2 రోజుల నుండి", "fever")

    def test_fever_bengali(self):
        self._assert_symptom_detected("জ্বর আছে ২ দিন ধরে", "fever")

    def test_fever_tamil(self):
        self._assert_symptom_detected("காய்ச்சல் உள்ளது 2 நாட்களுக்கு மேல்", "fever")

    def test_fever_gujarati(self):
        self._assert_symptom_detected("તાપ છે 2 દિવસથી", "fever")

    def test_fever_kannada(self):
        self._assert_symptom_detected("ಜ್ವರ ಬಂದಿದೆ 2 ದಿನದಿಂದ", "fever")

    def test_fever_malayalam(self):
        self._assert_symptom_detected("ജ്വരം ഉണ്ട് 2 ദിവസത്തിനുള്ളത്", "fever")

    def test_fever_odia(self):
        self._assert_symptom_detected("ଜ୍ୱର ଅଛି 2 ଦିନ ଧରି", "fever")

    def test_fever_punjabi(self):
        self._assert_symptom_detected("ਬੁਖਾਰ ਹੈ 2 ਦਿਨ ਤੋਂ", "fever")

    def test_fever_assamese(self):
        self._assert_symptom_detected("জ্বৰ আছে 2 দিনৰ পৰা", "fever")

    # --- HEADACHE ---
    def test_headache_english(self):
        self._assert_symptom_detected("Severe headache and pain", "headache")

    def test_headache_hindi(self):
        self._assert_symptom_detected("सिरदर्द है बहुत ज्यादा", "headache")

    def test_headache_marathi(self):
        self._assert_symptom_detected("डोकेदुखी आहे तीव्र", "headache")

    def test_headache_telugu(self):
        self._assert_symptom_detected("తలనొప్పి ఉంది తీవ్రంగా", "headache")

    def test_headache_bengali(self):
        self._assert_symptom_detected("মাথা ব্যথা হয়েছে খুব তীব্র", "headache")

    def test_headache_tamil(self):
        self._assert_symptom_detected("தலைவலி கடுமையாக உள்ளது", "headache")

    def test_headache_gujarati(self):
        self._assert_symptom_detected("માથું દુખાવો છે ખૂબ જ ટીવ્ર", "headache")

    def test_headache_kannada(self):
        self._assert_symptom_detected("ತಲೆ ನೋವು ಕಠಿಣವಾಗಿದೆ", "headache")

    def test_headache_malayalam(self):
        self._assert_symptom_detected("തലവേദന കഠിനമാണ്", "headache")

    def test_headache_odia(self):
        self._assert_symptom_detected("ମୁଣ୍ଡ ଦୁଃଖ ଅଛି ତୀବ୍ର", "headache")

    def test_headache_punjabi(self):
        self._assert_symptom_detected("ਸਿਰ ਦਰਦ ਹੈ ਬਹੁਤ ਤੀਬਰ", "headache")

    def test_headache_assamese(self):
        self._assert_symptom_detected("মূথ আঠা হৈছে অতি তীব্ৰ", "headache")

    # --- DIZZINESS ---
    def test_dizziness_english(self):
        self._assert_symptom_detected("Patient feels dizzy when standing", "dizziness")

    def test_dizziness_hindi(self):
        self._assert_symptom_detected("चक्कर आ रहे हैं खड़े होते समय", "dizziness")

    def test_dizziness_bengali(self):
        self._assert_symptom_detected("উপস্থিত থাকলে মাথা ঘোরা যায়", "dizziness")

    def test_dizziness_tamil(self):
        self._assert_symptom_detected("நின்றபோது தலை சுழற்சி உள்ளது", "dizziness")

    def test_dizziness_gujarati(self):
        self._assert_symptom_detected("ઉભા થતાં ચક્કર આવે છે", "dizziness")

    def test_dizziness_kannada(self):
        self._assert_symptom_detected("ನಿಂತ್ರು ಚಕ್ರ ಬರುತ್ತದೆ", "dizziness")

    def test_dizziness_malayalam(self):
        self._assert_symptom_detected("നിൽക്കുമ്പോൾ തല സഞ്ചരിക്കുന്നു", "dizziness")

    def test_dizziness_odia(self):
        self._assert_symptom_detected("ଉଠିବା ସମୟରେ ମୁଣ୍ଡ ଘୁର୍ମା ଆସେ", "dizziness")

    def test_dizziness_punjabi(self):
        self._assert_symptom_detected("ਖੜ੍ਹੇ ਹੋਣ 'ਤੇ ਸਿਰ ਘੁਮਦਾ ਹੈ", "dizziness")

    def test_dizziness_assamese(self):
        self._assert_symptom_detected("মূথ ঘূৰা যাওা", "dizziness")

    # --- SWELLING ---
    def test_swelling_english(self):
        self._assert_symptom_detected("Swelling in legs and feet", "swelling")

    def test_swelling_hindi(self):
        self._assert_symptom_detected("पैरों में सूजन है", "swelling")

    def test_swelling_bengali(self):
        self._assert_symptom_detected("পায়ে স্ফীতি আছে", "swelling")

    def test_swelling_tamil(self):
        self._assert_symptom_detected("காலில் வீக்கம் உள்ளது", "swelling")

    def test_swelling_gujarati(self):
        self._assert_symptom_detected("પગમાં સૂજ છે", "swelling")

    def test_swelling_kannada(self):
        self._assert_symptom_detected("ಕಾಲಿನಲ್ಲಿ ಊರು ಇದೆ", "swelling")

    def test_swelling_malayalam(self):
        self._assert_symptom_detected("കാലിൽ വീക്ക് ഉണ്ട്", "swelling")

    def test_swelling_odia(self):
        self._assert_symptom_detected("ପାଦରେ ସୂଜ ଅଛି", "swelling")

    def test_swelling_punjabi(self):
        self._assert_symptom_detected("ਲੱਕੀਰਾਂ ਵਿੱਚ ਸੂਜ ਹੈ", "swelling")

    def test_swelling_assamese(self):
        self._assert_symptom_detected("লগতৰ সূজ আছে", "swelling")

    # --- VOMITING ---
    def test_vomiting_english(self):
        self._assert_symptom_detected("Patient is vomiting frequently", "vomiting")

    def test_vomiting_hindi(self):
        self._assert_symptom_detected("बार-बार उल्टी हो रही है", "vomiting")

    def test_vomiting_bengali(self):
        self._assert_symptom_detected("বমি ঘন ঘন হচ্ছে", "vomiting")

    def test_vomiting_tamil(self):
        self._assert_symptom_detected("வாந்தி அடிக்கடி வருகிறது", "vomiting")

    def test_vomiting_gujarati(self):
        self._assert_symptom_detected("વાંતી વારંવાર આવે છે", "vomiting")

    def test_vomiting_kannada(self):
        self._assert_symptom_detected("ವಾಂತಿ ಬಾರಂಬಾರ ಆಗುತ್ತದೆ", "vomiting")

    def test_vomiting_malayalam(self):
        self._assert_symptom_detected("വാന്തി പലപ്പെട്ടു വരുന്നു", "vomiting")

    def test_vomiting_odia(self):
        self._assert_symptom_detected("ବାନ୍ତି ବାରମ୍ବାର ହୁଏ", "vomiting")

    def test_vomiting_punjabi(self):
        self._assert_symptom_detected("ਵੰਤੀ ਬਾਰ-ਬਾਰ ਆ ਰਹੀ ਹੈ", "vomiting")

    def test_vomiting_assamese(self):
        self._assert_symptom_detected("বমি পুনৰাপুনৰ্য হৈছে", "vomiting")

    # --- BLEEDING ---
    def test_bleeding_english(self):
        self._assert_symptom_detected("There is bleeding from the vagina", "bleeding")

    def test_bleeding_hindi(self):
        self._assert_symptom_detected("खून आ रहा है योनि से", "bleeding")

    def test_bleeding_bengali(self):
        self._assert_symptom_detected("যোনি থেকে রক্তপাত হচ্ছে", "bleeding")

    def test_bleeding_tamil(self):
        self._assert_symptom_detected("யோனியில் இருந்து இரத்தம் விழுகிறது", "bleeding")

    def test_bleeding_gujarati(self):
        self._assert_symptom_detected("યોનિ પરથી લોબો વટી રહ્યું છે", "bleeding")

    def test_bleeding_kannada(self):
        self._assert_symptom_detected("ಯೋನಿಯಿಂದ ರಕ್ತ ಹರಿಯುತ್ತಿದೆ", "bleeding")

    def test_bleeding_malayalam(self):
        self._assert_symptom_detected("യോനിയിൽ നിന്ന് രക്തപാതം ഉണ്ട്", "bleeding")

    def test_bleeding_odia(self):
        self._assert_symptom_detected("ଯୋନି ରୁ ରକ୍ତସ୍ରାବ ହେଉଛି", "bleeding")

    def test_bleeding_punjabi(self):
        self._assert_symptom_detected("ਯੋਨੀ ਤੋਂ ਖੂਨ ਵਟ ਰਿਹਾ ਹੈ", "bleeding")

    def test_bleeding_assamese(self):
        self._assert_symptom_detected("যোনিৰ পৰা ৰক্তপাত হৈছে", "bleeding")

    # --- WEAKNESS ---
    def test_weakness_english(self):
        self._assert_symptom_detected("Patient feels very weak and tired", "weakness")

    def test_weakness_hindi(self):
        self._assert_symptom_detected("बहुत कमजोरी महसूस हो रही है", "weakness")

    def test_weakness_bengali(self):
        self._assert_symptom_detected("অনেক দুর্বলতাই অনুভব করছে", "weakness")

    def test_weakness_tamil(self):
        self._assert_symptom_detected("ரோଗி மிகவும் பலவின்மை உணர்கிறது", "weakness")

    def test_weakness_gujarati(self):
        self._assert_symptom_detected("રોગી ખૂબ જ કમજોરી અનુભવે છે", "weakness")

    def test_weakness_kannada(self):
        self._assert_symptom_detected("ರೋಗಿ ತುಂಬಾ ದುರ್ಬಲತೆ ಅನುಭವಿಸುತ್ತಿದ್ದಾನೆ", "weakness")

    def test_weakness_malayalam(self):
        self._assert_symptom_detected("രോഗി അത്യന്തം ദുർബലത അനുഭവിക്കുന്നു", "weakness")

    def test_weakness_odia(self):
        self._assert_symptom_detected("ରୋଗୀ ବହୁତ ଦୁର୍ବଳତା ଅନୁଭବ କରୁଛି", "weakness")

    def test_weakness_punjabi(self):
        self._assert_symptom_detected("ਰੋਗੀ ਬਹੁਤ ਕਮਜ਼ੋਰੀ ਮਹਿਸੂਸ ਕਰਦਾ ਹੈ", "weakness")

    def test_weakness_assamese(self):
        self._assert_symptom_detected("ৰোগী অতি দुৰ୍ବଳତା অনুভৱ কৰে", "weakness")

    # --- DIARRHOEA ---
    def test_diarrhoea_english(self):
        self._assert_symptom_detected("Patient has diarrhoea since morning", "diarrhoea")

    def test_diarrhoea_hindi(self):
        self._assert_symptom_detected("दस्त है सुबह से", "diarrhoea")

    def test_diarrhoea_bengali(self):
        self._assert_symptom_detected("ডায়রিয়া আছে সকাল থেকে", "diarrhoea")

    def test_diarrhoea_tamil(self):
        self._assert_symptom_detected("வயிற்றுப்புணர் காலையிலிருந்து உள்ளது", "diarrhoea")

    def test_diarrhoea_gujarati(self):
        self._assert_symptom_detected("ડાયરિયા છે સવારથી", "diarrhoea")

    def test_diarrhoea_kannada(self):
        self._assert_symptom_detected("ಅತಿಸಾರ ಬೆಳಗಿನದ Clint underscore ಸಮಯದಿಂದ", "diarrhoea")

    def test_diarrhoea_malayalam(self):
        self._assert_symptom_detected("അതിസാരം ഉണ്ട് രാത്രി മുതൽ", "diarrhoea")

    def test_diarrhoea_odia(self):
        self._assert_symptom_detected("ଅତିସାର ଅଛି ସକାଳି ଠାରୁ", "diarrhoea")

    def test_diarrhoea_punjabi(self):
        self._assert_symptom_detected("ਦਸਤ ਹੈ ਸਵੇਰੇ ਤੋਂ", "diarrhoea")

    def test_diarrhoea_assamese(self):
        self._assert_symptom_detected("দস্ত আছে পূৰ্বাহ্ণৰ পৰা", "diarrhoea")

    # --- BLURRED VISION ---
    def test_blurred_vision_english(self):
        self._assert_symptom_detected("Patient has blurred vision", "blurred vision")

    def test_blurred_vision_hindi(self):
        self._assert_symptom_detected("धुंधला दिखाई देना है", "blurred vision")

    def test_blurred_vision_bengali(self):
        self._assert_symptom_detected("দৃষ্টিভঙ্গি হয়ে উঠেছে", "blurred vision")

    def test_blurred_vision_tamil(self):
        self._assert_symptom_detected("கண் தெரியாது மங்கிய காட்சி", "blurred vision")

    def test_blurred_vision_gujarati(self):
        self._assert_symptom_detected("ધૂમ્રેલું દેખાવ છે", "blurred vision")

    def test_blurred_vision_kannada(self):
        self._assert_symptom_detected("ದೃಷ್ಟಿ ಮಸಕು ಆಗಿದೆ", "blurred vision")

    def test_blurred_vision_malayalam(self):
        self._assert_symptom_detected("കാഴ്ച മങ്ങിയത്", "blurred vision")

    def test_blurred_vision_odia(self):
        self._assert_symptom_detected("ଦୃଷ୍ଟି ଧୂମ୍ର ହୋଇଛି", "blurred vision")

    def test_blurred_vision_punjabi(self):
        self._assert_symptom_detected("ਦਿੱਖ ਧੂਮ੍ਰੀ ਹੋ ਰਹੀ ਹੈ", "blurred vision")

    def test_blurred_vision_assamese(self):
        self._assert_symptom_detected("ধুম্ৰে দৃষ্টি হৈছে", "blurred vision")

    # --- SEVERE ABDOMINAL PAIN ---
    def test_severe_abdominal_pain_english(self):
        self._assert_symptom_detected("Severe stomach pain in lower abdomen", "severe abdominal pain")

    def test_severe_abdominal_pain_hindi(self):
        self._assert_symptom_detected("तेज पेट दर्द है निचले पेट में", "severe abdominal pain")

    def test_severe_abdominal_pain_bengali(self):
        self._assert_symptom_detected("তীব্র পেট ব্যথা নিচের পেটে", "severe abdominal pain")

    def test_severe_abdominal_pain_tamil(self):
        self._assert_symptom_detected("கடுமையான வயிற்று வலி கீழ்நிற்கில்", "severe abdominal pain")

    def test_severe_abdominal_pain_gujarati(self):
        self._assert_symptom_detected("તીવ્ર પેટ દુખાવો નીચે પેટમાં", "severe abdominal pain")

    def test_severe_abdominal_pain_kannada(self):
        self._assert_symptom_detected("ಕಠಿಣ ಹೊಟ್ಟೆ ನೋವು ಕೆಳಗೆ ಹೊಟ್ಟೆಯಲ್ಲಿ", "severe abdominal pain")

    def test_severe_abdominal_pain_malayalam(self):
        self._assert_symptom_detected("കടുത്ത വയ നോവു താഴെ വയിൽ", "severe abdominal pain")

    def test_severe_abdominal_pain_odia(self):
        self._assert_symptom_detected("ତୀବ୍ର ପେଟ ଦୁଃଖ ନିଚ ପେଟରେ", "severe abdominal pain")

    def test_severe_abdominal_pain_punjabi(self):
        self._assert_symptom_detected("ਤੀਬਰ ਪੇਟ ਦard ਹੇਠੋਂ ਪੇਟ ਵਿੱਚ", "severe abdominal pain")

    def test_severe_abdominal_pain_assamese(self):
        self._assert_symptom_detected("তীব্ৰ পেটৰ দুঃখ তলৰ পেটত", "severe abdominal pain")

    # --- REDUCED FETAL MOVEMENT ---
    def test_reduced_fetal_movement_english(self):
        self._assert_symptom_detected("Baby moving less than usual", "reduced fetal movement")

    def test_reduced_fetal_movement_hindi(self):
        self._assert_symptom_detected("बाळाची हालचाल कमी होती आहे", "reduced fetal movement")

    def test_reduced_fetal_movement_bengali(self):
        self._assert_symptom_detected("শিশুর গতি কম হয়েছে", "reduced fetal movement")

    def test_reduced_fetal_movement_tamil(self):
        self._assert_symptom_detected("குழந்தை நகர்வு குறைந்தது", "reduced fetal movement")

    def test_reduced_fetal_movement_gujarati(self):
        self._assert_symptom_detected("બાળકની હાલત ઓછી થઈ રહી છે", "reduced fetal movement")

    def test_reduced_fetal_movement_kannada(self):
        self._assert_symptom_detected("ಮಗುವಿನ ಚಲನೆ ಕಡಿಮೆ ಆಗಿದೆ", "reduced fetal movement")

    def test_reduced_fetal_movement_malayalam(self):
        self._assert_symptom_detected("ശിശുവിന്റെ ചലനം കുറഞ്ഞു", "reduced fetal movement")

    def test_reduced_fetal_movement_odia(self):
        self._assert_symptom_detected("ଶିଶୁ ଚଳନ କମ୍ ହୋଇଛି", "reduced fetal movement")

    def test_reduced_fetal_movement_punjabi(self):
        self._assert_symptom_detected("ਬੱਚਾ ਘੱਟ ਹਿਲ ਰਿਹਾ ਹੈ", "reduced fetal movement")

    def test_reduced_fetal_movement_assamese(self):
        self._assert_symptom_detected("বাচ্চা অল্প চলছে", "reduced fetal movement")


class PregnancyMonthExtractionTests(unittest.TestCase):
    def test_month_extraction_english(self):
        result = process_healthcare_input("Patient is 7 months pregnant")
        self.assertEqual(result["pregnancy_month"], "7")

    def test_month_extraction_hindi(self):
        result = process_healthcare_input("गर्भवती महिना 8 है")
        self.assertEqual(result["pregnancy_month"], "8")

    def test_month_extraction_bengali(self):
        result = process_healthcare_input("গর্ভবতী 7 মাস")
        self.assertEqual(result["pregnancy_month"], "7")

    def test_month_extraction_tamil(self):
        result = process_healthcare_input("கர்ப்பமுள்ளவர் 5 மாதம்")
        self.assertEqual(result["pregnancy_month"], "5")

    def test_month_extraction_gujarati(self):
        result = process_healthcare_input("ગર્ભવતી 6 મહિનો")
        self.assertEqual(result["pregnancy_month"], "6")

    def test_month_extraction_kannada(self):
        result = process_healthcare_input("ಗರ್ಭವತಿ 4 ತಿಂಗಳು")
        self.assertEqual(result["pregnancy_month"], "4")

    def test_month_extraction_malayalam(self):
        result = process_healthcare_input("ഗർഭവതി 3 മാസം")
        self.assertEqual(result["pregnancy_month"], "3")

    def test_month_extraction_odia(self):
        result = process_healthcare_input("ଗର୍ଭବତୀ 9 ମାସ")
        self.assertEqual(result["pregnancy_month"], "9")

    def test_month_extraction_punjabi(self):
        result = process_healthcare_input("ਗਰਭਵਤੀ 2 ਮਹੀਨਾ")
        self.assertEqual(result["pregnancy_month"], "2")

    def test_month_extraction_assamese(self):
        result = process_healthcare_input("গাৰ্ভবতী 1 মাহ")
        self.assertEqual(result["pregnancy_month"], "1")


class NegationTests(unittest.TestCase):
    def _assert_symptom_not_detected(self, text: str, symptom: str) -> None:
        result = process_healthcare_input(text)
        self.assertNotIn(symptom, result["symptoms"],
                         f"Did not expect '{symptom}' in {result['symptoms']} for input: {text}")

    def test_bengali_negation_excludes_fever(self):
        self._assert_symptom_not_detected("জ্বর নয়", "fever")

    def test_tamil_negation_excludes_fever(self):
        self._assert_symptom_not_detected("காய்ச்சல் இல்லை", "fever")

    def test_gujarati_negation_excludes_fever(self):
        self._assert_symptom_not_detected("તાપ નથી", "fever")

    def test_kannada_negation_excludes_fever(self):
        self._assert_symptom_not_detected("ಜ್ವರ ಇಲ್ಲ", "fever")

    def test_malayalam_negation_excludes_fever(self):
        self._assert_symptom_not_detected("ജ്വരം ഇല്ല", "fever")

    def test_odia_negation_excludes_fever(self):
        self._assert_symptom_not_detected("ଜ୍ୱର ନାହିଁ", "fever")

    def test_punjabi_negation_excludes_fever(self):
        self._assert_symptom_not_detected("ਬੁਖਾਰ ਨਹੀਂ ਹੈ", "fever")

    def test_assamese_negation_excludes_fever(self):
        self._assert_symptom_not_detected("জ্বৰ নাই", "fever")


class RiskClassificationTests(unittest.TestCase):
    def test_high_risk_bengali_bleeding(self):
        result = process_healthcare_input("রক্তপাত হচ্ছে")
        self.assertEqual(result["risk_level"], "HIGH RISK")

    def test_high_risk_tamil_severe_pain(self):
        result = process_healthcare_input("கடுமையான வயிற்று வலி")
        self.assertEqual(result["risk_level"], "HIGH RISK")

    def test_medium_risk_gujarati_fever(self):
        result = process_healthcare_input("તાપ છે")
        self.assertEqual(result["risk_level"], "MEDIUM RISK")

    def test_medium_risk_malayalam_vomiting(self):
        result = process_healthcare_input("വാന്തി വരുന്നു")
        self.assertEqual(result["risk_level"], "MEDIUM RISK")

    def test_low_risk_punjabi_routine(self):
        result = process_healthcare_input("ਰੋਜ਼ਾਨਾ ਜਾਂਚ")
        self.assertEqual(result["risk_level"], "LOW RISK")

    def test_high_risk_assamese_reduced_movement(self):
        result = process_healthcare_input("শিশুৰ গতি কম")
        self.assertEqual(result["risk_level"], "HIGH RISK")


class RegressionTests(unittest.TestCase):
    def test_english_still_works(self):
        result = process_healthcare_input("Patient has fever and headache")
        self.assertIn("fever", result["symptoms"])
        self.assertIn("headache", result["symptoms"])

    def test_hindi_still_works(self):
        result = process_healthcare_input("बुखार और सिरदर्द है")
        self.assertIn("fever", result["symptoms"])
        self.assertIn("headache", result["symptoms"])

    def test_marathi_still_works(self):
        result = process_healthcare_input("ताप आला आहे आणि डोकेदुखी")
        self.assertIn("fever", result["symptoms"])
        self.assertIn("headache", result["symptoms"])

    def test_telugu_still_works(self):
        result = process_healthcare_input("జ్వరం ఉంది తలనొప్పి ఉంది")
        self.assertIn("fever", result["symptoms"])
        self.assertIn("headache", result["symptoms"])


if __name__ == "__main__":
    unittest.main()
