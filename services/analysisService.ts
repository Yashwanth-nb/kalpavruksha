import { GoogleGenAI, Type } from '@google/genai';
import { PredictionResult, Coordinates, Expert, Language } from '../types';

// Use Vite's import.meta.env instead of process.env
const GEMINI_API_KEY = import.meta.env.VITE_GEMINI_API_KEY;
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

// Debug logs
console.log('🔗 API URL:', API_URL);
console.log('🔑 Gemini API Key:', GEMINI_API_KEY ? '✅ Set' : '❌ Not set');

if (!GEMINI_API_KEY) {
  console.error('❌ VITE_GEMINI_API_KEY is not set in environment variables!');
}

const ai = new GoogleGenAI({ apiKey: GEMINI_API_KEY || '' });

const diseaseAnalysisSchema = {
  type: Type.OBJECT,
  properties: {
    isHealthy: {
      type: Type.BOOLEAN,
      description: 'True if the plant is healthy, false otherwise.',
    },
    diseaseType: {
      type: Type.STRING,
      description: 'The specific name of the disease detected (e.g., "Bud Rot", "Leaf Blight", "Stem Bleeding"). If healthy, this should be "Healthy". If a disease is present but unidentifiable, use "Unknown".',
    },
    severity: {
      type: Type.STRING,
      enum: ['Mild', 'Moderate', 'Severe', 'N/A'],
      description: 'The severity of the disease. "N/A" if healthy.',
    },
    confidence: {
      type: Type.NUMBER,
      description: 'A confidence score from 0.0 to 1.0 for the entire prediction.',
    },
  },
  required: ['isHealthy', 'diseaseType', 'severity', 'confidence'],
};

const expertsSchema = {
    type: Type.ARRAY,
    items: {
      type: Type.OBJECT,
      properties: {
        name: {
          type: Type.STRING,
          description: 'The name of the agricultural expert or organization.',
        },
        address: {
          type: Type.STRING,
          description: 'The physical address of the expert or organization.',
        },
        phone: {
          type: Type.STRING,
          description: 'The contact phone number.',
        },
      },
      required: ['name', 'address', 'phone'],
    },
};

export const analyzeWithGemini = async (base64Image: string, mimeType: string) => {
  const imagePart = {
    inlineData: {
      data: base64Image,
      mimeType: mimeType,
    },
  };

  const textPart = {
    text: `You are a specialized machine learning model, analogous to a Convolutional Neural Network (CNN), trained on a vast dataset of coconut tree pathology images. Your function is to perform high-accuracy disease detection.

    Analyze the provided image of a coconut tree component (leaf, stem, bud, etc.).

    Follow these steps precisely:
    1.  **Image Classification:** Determine if the subject is 'Healthy' or 'Diseased'.
    2.  **Disease Identification:** If diseased, classify it into one of the following categories: "Bud Rot", "Leaf Blight", "Stem Bleeding", or "Unknown" if the symptoms are ambiguous or not from a common disease.
    3.  **Severity Estimation:** Quantify the disease progression as "Mild", "Moderate", or "Severe". If the tree is healthy, this should be "N/A".
    4.  **Confidence Score:** Provide a confidence score (from 0.0 to 1.0) representing the certainty of your combined diagnosis (health status, disease type, and severity).

    Your output MUST be a single, clean JSON object matching the provided schema. Do not include any explanatory text or markdown.`,
  };

  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash',
    contents: { parts: [imagePart, textPart] },
    config: {
      responseMimeType: "application/json",
      responseSchema: diseaseAnalysisSchema,
    }
  });

  try {
    const jsonString = response.text.trim();
    const result = JSON.parse(jsonString);
    return result;
  } catch (e) {
    console.error("Failed to parse Gemini response:", response.text);
    throw new Error("AI response was not in the expected format.");
  }
};

export const analyzeWithCustomModel = async (imageFile: File): Promise<{ prediction: string; confidence: number }> => {
  const formData = new FormData();
  formData.append('file', imageFile);

  try {
    console.log('📤 Sending request to:', `${API_URL}/predict`);
    
    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("Backend error:", errorText);
      throw new Error(`Request failed with status ${response.status}: ${errorText}`);
    }

    const result = await response.json();
    console.log('✅ Analysis result:', result);
    return result;
  } catch (error) {
    console.error('❌ Error calling custom model:', error);
    throw error;
  }
};

export const getTreatmentRecommendation = async (diseaseType: string, lang: Language): Promise<string> => {
  const normalizeKey = (s: string) =>
    (s || '')
      .toLowerCase()
      .replace('caterpillers', 'caterpillars')
      .replace(/[^a-z0-9]/g, '');
  const languageMap: Record<Language, string> = {
    [Language.ENGLISH]: 'English',
    [Language.KANNADA]: 'Kannada',
    [Language.TAMIL]: 'Tamil',
    [Language.TELUGU]: 'Telugu',
    [Language.MALAYALAM]: 'Malayalam',
  };
  const languageName = languageMap[lang] || 'English';

  const productLinks: Record<string, string | string[] | { name: string; url?: string } | Array<{ name: string; url?: string }>> = {
    'bud root dropping': 'https://cultree.in/products/shivalik-roksin-thiophanate-methyl-70-wp-fungicide',
    'bud rot': 'https://cultree.in/products/shivalik-roksin-thiophanate-methyl-70-wp-fungicide',
    'gray leaf spot': 'https://www.bighaat.com/products/agriventure-cooxy',
    'leaf rot': 'https://ariesagro.com/jahaan-hexaconazole5-w-w/',
    'stembleeding': [
      "https://www.bighaat.com/products/blue-copper-fungicide-1",
      "http://www.rayfull.com/Productshows.asp?ID=338",
      "https://krishisevakendra.in/products/bordeaux-mixture"
    ],
    'caterpillars': { name: 'Phoskill Insecticide-(Upl)' },
    'drying': [
      { name: 'Potassium Nitrate (KNO3)' },
      { name: 'THE WET TREE Bio Larvicide (Bacillus Thuringiensis Var Kurstaki) Liquid | Bio Pesticide | Insecticide for Plants 500ml', url: ' https://amzn.in/d/0FqYTFA ' }
    ],
    'flaccidity': [
      { name: 'Zinc + Boron foliar spray' },
      { name: 'Paraquat Dichloride 24% SL', url: ' https://amzn.in/d/ghZd5LX ' }
    ],
    'leaflet': { name: 'Copper oxychloride fungicide' , url: 'https://share.google/t1QmVne4AGQHQ4JSB' },
    'yellowing': [
      { name: 'Ferrous sulphate (Fe) correction' },
      { name: 'Bacillus Thuringiensis Var Kurstaki', url: ' https://amzn.in/d/eYa0DT8 ' }
    ]
  };

  const getStaticProducts = (): Array<{ name: string; url?: string }> => {
    const dNorm = normalizeKey(diseaseType);
    const key = Object.keys(productLinks).find(k => {
      const kNorm = normalizeKey(k);
      return dNorm.includes(kNorm) || kNorm.includes(dNorm);
    });
    if (!key) return [];
    const asList = (val: any): Array<{ name: string; url?: string }> => {
      if (Array.isArray(val)) return val as Array<{ name: string; url?: string }>;
      if (typeof val === 'string') return [{ name: 'Product', url: val }];
      if (val && typeof val === 'object') return [val as { name: string; url?: string }];
      return [];
    };
    return asList((productLinks as any)[key]);
  };

  const renderMarkdown = (items: Array<{ name: string; url?: string }>) => {
    if (!items.length) return '';
    const lines = items.map((it, i) => it.url ? `- [${(it.name || '').trim()}](${(it.url || '').trim()})` : `- ${(it.name || '').trim()}`).join('\n');
    return `\n\n**Recommended Products:**\n${lines}`;
  };

  let productMarkdown = '';
  try {
    const dNorm = normalizeKey(diseaseType);
    let merged: any = { items: [] };

    const resp = await fetch('/data/products.json');
    if (resp.ok) {
      const data = await resp.json();
      merged.items = Array.isArray(data.items) ? data.items : [];
    }

    let lsProducts: Array<{ name: string; url?: string }> = [];
    try {
      const raw = localStorage.getItem('cg-products');
      if (raw) {
        const ls = JSON.parse(raw);
        const lsItems = Array.isArray(ls.items) ? ls.items : [];
        lsItems.forEach((lsItem: any) => {
          const key = String(lsItem.key);
          const keyNorm = normalizeKey(key);
          const idx = merged.items.findIndex((it: any) => normalizeKey(String(it.key)) === keyNorm);
          if (idx >= 0) {
            const base = Array.isArray(merged.items[idx].products) ? merged.items[idx].products : [];
            merged.items[idx].products = base.concat(lsItem.products || []);
          } else {
            merged.items.push(lsItem);
          }
        });
        const entryLs = (merged.items || []).find((it: any) => normalizeKey(String(it.key)) === dNorm || dNorm.includes(normalizeKey(String(it.key))) || normalizeKey(String(it.key)).includes(dNorm));
        if (entryLs && Array.isArray(entryLs.products)) {
          lsProducts = entryLs.products as Array<{ name: string; url?: string }>;
        }
      }
    } catch {}
    
    const staticProducts = getStaticProducts();
    const entryJson = (merged.items || []).find((it: any) => {
      const kNorm = normalizeKey(String(it.key));
      return dNorm.includes(kNorm) || kNorm.includes(dNorm);
    });
    const jsonProducts: Array<{ name: string; url?: string }> = entryJson && Array.isArray(entryJson.products) ? entryJson.products : [];

    const combined = [...lsProducts, ...staticProducts, ...jsonProducts]
      .map(p => ({ name: (p?.name || '').trim(), url: p?.url ? String(p.url).trim() : undefined }))
      .filter(p => p.name);
    const seen = new Set<string>();
    const unique = combined.filter(p => {
      const key = `${p.name}|${p.url || ''}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
    productMarkdown = renderMarkdown(unique);
  } catch (e) {
    // ignore
  }
  if (!productMarkdown) {
    productMarkdown = renderMarkdown(getStaticProducts());
  }

  const prompt = `Provide a concise, actionable treatment plan for a coconut tree diagnosed with "${diseaseType}". The plan should be easy for a farmer to follow. Include both chemical and organic/cultural management strategies if possible. The response MUST be in the ${languageName} language.
  ${productMarkdown ? `\nAt the end of the recommendation, include the following recommended product section exactly as written, without adding extra text or altering the format:\n${productMarkdown}` : ''}
  `;

  const maxRetries = 3;
  for (let i = 0; i < maxRetries; i++) {
    try {
      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: prompt,
      });
      return response.text.trim();
    } catch (error) {
      console.error(`Attempt ${i + 1} failed:`, error);
      if (i === maxRetries - 1) {
        return "Could not retrieve treatment recommendation at this time due to server overload. Please try again later.";
      }
      await new Promise(res => setTimeout(res, 1000 * Math.pow(2, i)));
    }
  }
  return "Could not retrieve treatment recommendation at this time.";
};

export const findLocalExperts = async (coordinates: Coordinates, diseaseInfo: string, lang: Language): Promise<Expert[]> => {
    const languageMap: Record<Language, string> = {
      [Language.ENGLISH]: 'English',
      [Language.KANNADA]: 'Kannada',
      [Language.TAMIL]: 'Tamil',
      [Language.TELUGU]: 'Telugu',
      [Language.MALAYALAM]: 'Malayalam',
    };
    const languageName = languageMap[lang] || 'English';

    const prompt = `Based on the location (Latitude: ${coordinates.latitude}, Longitude: ${coordinates.longitude}), find 3-5 local agricultural extension offices, university agricultural departments, or certified agronomists who specialize in tropical plants or coconut palms. The user is dealing with this issue: "${diseaseInfo}". Provide their name, address, and a contact phone number. Respond in ${languageName} where appropriate (e.g., for general descriptions), but keep names and addresses in their original form.`;

    const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: prompt,
        config: {
            responseMimeType: "application/json",
            responseSchema: expertsSchema,
        }
    });

    try {
        const jsonString = response.text.trim();
        const result: Expert[] = JSON.parse(jsonString);
        return result;
    } catch (e) {
        console.error("Failed to parse experts response:", response.text);
        throw new Error("AI response for experts was not in the expected format.");
    }
};

export const getAIAssistantResponse = async (prompt: string, lang: Language): Promise<string> => {
  const languageMap: Record<Language, string> = {
    [Language.ENGLISH]: 'English',
    [Language.KANNADA]: 'Kannada',
    [Language.TAMIL]: 'Tamil',
    [Language.TELUGU]: 'Telugu',
    [Language.MALAYALAM]: 'Malayalam',
  };
  const languageName = languageMap[lang] || 'English';

  const fullPrompt = `You are a helpful AI assistant specializing in agriculture, particularly for coconut farming. The user has a question. Provide a helpful, concise, and easy-to-understand answer. The response MUST be in the ${languageName} language.

User's question: "${prompt}"`;

  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash',
    contents: fullPrompt,
  });

  return response.text;
};

export const getPlaceName = async (lat: number, lon: number): Promise<string> => {
  try {
    const response = await fetch(`https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=${lat}&lon=${lon}`);
    if (!response.ok) {
      console.error('Failed to fetch place name from Nominatim');
      return 'N/A';
    }
    const data = await response.json();
    return data.display_name || 'N/A';
  } catch (error) {
    console.error('Error in reverse geocoding:', error);
    return 'N/A';
  }
};