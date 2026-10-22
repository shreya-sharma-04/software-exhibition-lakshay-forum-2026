import streamlit as st
import joblib
import pandas as pd
import requests
import re
import ipaddress
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from collections import Counter
from difflib import SequenceMatcher

MODEL_PATH = "phishing_detection_model.pkl"

FEATURES = [
    "URLLength", "DomainLength", "IsDomainIP",
    "URLSimilarityIndex", "CharContinuationRate",
    "TLDLegitimateProb", "URLCharProb", "TLDLength",
    "NoOfSubDomain", "HasObfuscation",
    "NoOfObfuscatedChar", "ObfuscationRatio",
    "NoOfLettersInURL", "LetterRatioInURL",
    "NoOfDegitsInURL", "DegitRatioInURL",
    "NoOfEqualsInURL", "NoOfQMarkInURL",
    "NoOfAmpersandInURL", "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL", "IsHTTPS", "LineOfCode",
    "LargestLineLength", "HasTitle", "DomainTitleMatchScore",
    "URLTitleMatchScore", "HasFavicon", "Robots",
    "IsResponsive", "NoOfURLRedirect", "NoOfSelfRedirect",
    "HasDescription", "NoOfPopup", "NoOfiFrame",
    "HasExternalFormSubmit", "HasSocialNet", "HasSubmitButton",
    "HasHiddenFields", "HasPasswordField", "Bank", "Pay",
    "Crypto", "HasCopyrightInfo", "NoOfImage", "NoOfCSS",
    "NoOfJS", "NoOfSelfRef", "NoOfEmptyRef", "NoOfExternalRef"
]

st.set_page_config(
    page_title="AI Phishing Detection",
    page_icon="🛡️"
)


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)

    if getattr(model, "n_features_in_", None) != 50:
        raise ValueError("This app expects a 50-feature model.")

    return model


def safe_fetch(url):
    try:
        response = requests.get(
            url,
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0"},
            allow_redirects=True,
        )
        return response
    except requests.RequestException:
        return None


def extract_features(url):
    p = urlparse(url)
    domain = p.hostname or ""
    path = p.path or ""
    full = url.lower()

    try:
        ipaddress.ip_address(domain)
        is_ip = 1
    except ValueError:
        is_ip = 0

    labels = domain.split(".")
    tld = labels[-1] if len(labels) > 1 else ""
    letters = sum(c.isalpha() for c in url)
    digits = sum(c.isdigit() for c in url)
    specials = sum(not c.isalnum() for c in url)
    obfuscated = len(re.findall(r"%[0-9a-fA-F]{2}", url))
    transitions = sum(
        1 for a, b in zip(url, url[1:])
        if a.isalnum() != b.isalnum()
    )

    # Fetch page content only after validating the URL scheme.
    response = safe_fetch(url)
    html = response.text[:1_000_000] if response is not None else ""
    soup = BeautifulSoup(html, "html.parser") if html else None

    title = ""
    if soup and soup.title:
        title = soup.title.get_text(" ", strip=True)

    def count(tag):
        return len(soup.find_all(tag)) if soup else 0

    def has_meta(name):
        return bool(
            soup and soup.find(
                "meta", attrs={"name": re.compile(name, re.I)}
            )
        )

    links = soup.find_all("a", href=True) if soup else []
    external = 0
    self_refs = 0
    empty_refs = 0

    for link in links:
        href = link.get("href", "").strip()
        if not href or href == "#":
            empty_refs += 1
        elif href.startswith(("http://", "https://")):
            if urlparse(href).hostname == domain:
                self_refs += 1
            else:
                external += 1
        else:
            self_refs += 1

    forms = soup.find_all("form") if soup else []
    external_form = int(any(
        (f.get("action") or "").startswith(("http://", "https://"))
        and urlparse(f.get("action")).hostname != domain
        for f in forms
    ))

    title_match = SequenceMatcher(
        None, domain.lower().replace("www.", ""),
        title.lower().replace(" ", "")
    ).ratio() if title else 0.0

    words = ["bank", "pay", "crypto"]
    text = (url + " " + title).lower()

    features = {
        "URLLength": len(url),
        "DomainLength": len(domain),
        "IsDomainIP": is_ip,
        "URLSimilarityIndex": 0.0,
        "CharContinuationRate": (
            1 - transitions / max(len(url) - 1, 1)
        ),
        "TLDLegitimateProb": 0.5,
        "URLCharProb": 0.0,
        "TLDLength": len(tld),
        "NoOfSubDomain": max(len(labels) - 2, 0),
        "HasObfuscation": int(obfuscated > 0 or "@" in url),
        "NoOfObfuscatedChar": obfuscated,
        "ObfuscationRatio": obfuscated / max(len(url), 1),
        "NoOfLettersInURL": letters,
        "LetterRatioInURL": letters / max(len(url), 1),
        "NoOfDegitsInURL": digits,
        "DegitRatioInURL": digits / max(len(url), 1),
        "NoOfEqualsInURL": url.count("="),
        "NoOfQMarkInURL": url.count("?"),
        "NoOfAmpersandInURL": url.count("&"),
        "NoOfOtherSpecialCharsInURL": sum(
            url.count(c) for c in "@!#$%^&*()_+[]{}|\\;:'\",<> "
        ),
        "SpacialCharRatioInURL": specials / max(len(url), 1),
        "IsHTTPS": int(p.scheme == "https"),
        "LineOfCode": len(html.splitlines()),
        "LargestLineLength": max(
            (len(line) for line in html.splitlines()), default=0
        ),
        "HasTitle": int(bool(title)),
        "DomainTitleMatchScore": title_match,
        "URLTitleMatchScore": SequenceMatcher(
            None, path.lower(), title.lower()
        ).ratio() if title else 0.0,
        "HasFavicon": int(bool(
            soup and soup.find("link", rel=re.compile("icon", re.I))
        )),
        "Robots": int(safe_fetch(
            f"{p.scheme}://{p.netloc}/robots.txt"
        ) is not None),
        "IsResponsive": int(bool(
            soup and soup.find(
                "meta", attrs={"name": "viewport"}
            )
        )),
        "NoOfURLRedirect": len(response.history) if response else 0,
        "NoOfSelfRedirect": sum(
            1 for r in response.history
            if urlparse(r.url).hostname == domain
        ) if response else 0,
        "HasDescription": int(has_meta("description")),
        "NoOfPopup": 0,
        "NoOfiFrame": count("iframe"),
        "HasExternalFormSubmit": external_form,
        "HasSocialNet": int(any(
            x in html.lower()
            for x in ["facebook.com", "instagram.com",
                      "linkedin.com", "twitter.com"]
        )),
        "HasSubmitButton": int(bool(
            soup and soup.find(
                ["button", "input"],
                attrs={"type": re.compile("submit", re.I)}
            )
        )),
        "HasHiddenFields": int(bool(
            soup and soup.find(
                "input", attrs={"type": "hidden"}
            )
        )),
        "HasPasswordField": int(bool(
            soup and soup.find(
                "input", attrs={"type": "password"}
            )
        )),
        "Bank": int("bank" in text),
        "Pay": int("pay" in text or "payment" in text),
        "Crypto": int(any(
            x in text for x in ["crypto", "bitcoin", "wallet"]
        )),
        "HasCopyrightInfo": int("copyright" in html.lower()),
        "NoOfImage": count("img"),
        "NoOfCSS": len(soup.find_all("link", rel="stylesheet"))
            if soup else 0,
        "NoOfJS": count("script"),
        "NoOfSelfRef": self_refs,
        "NoOfEmptyRef": empty_refs,
        "NoOfExternalRef": external,
    }

    return pd.DataFrame(
        [[features[name] for name in FEATURES]],
        columns=FEATURES
    )


st.title("🛡️ AI-Powered Phishing Detection")
st.write("Enter a website URL to inspect it.")

st.warning(
    "Experimental demo only: several features are approximated. "
    "Do not rely on this result to decide whether a website is safe."
)

url = st.text_input(
    "Website URL",
    placeholder="https://example.com"
)

if st.button("Check URL"):
    if not url.strip():
        st.warning("Please enter a URL.")
    else:
        if "://" not in url:
            url = "https://" + url.strip()

        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            st.error("Please enter a valid HTTP or HTTPS URL.")
        else:
            try:
                with st.spinner("Inspecting URL and webpage..."):
                    model = load_model()
                    X = extract_features(url)

                    prediction = int(model.predict(X)[0])

                st.subheader("Model result")

                # Verify these class meanings against the training labels.
                if prediction == 0:
                    st.error(
                        "Model class 0 returned. Treat this site as "
                        "potentially suspicious until class meanings "
                        "are verified."
                    )
                elif prediction == 1:
                    st.success(
                        "Model class 1 returned. This does NOT guarantee "
                        "that the website is legitimate or safe."
                    )
                else:
                    st.warning(f"Model returned class: {prediction}")

            except Exception as e:
                st.error(f"Could not make prediction: {e}")
