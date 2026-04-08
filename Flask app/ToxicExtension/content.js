// content.js

const BAD_WORDS = [
  'damn', 'hell', 'shit', 'fuck', 'bitch', 'bastard', 'asshole', 'idiot', 'stupid', 'crap',
  'piss', 'dick', 'slut', 'whore', 'ugly', 'nasty', 'trash', 'suck', 'kill'
];

function countBadWords(text) {
  const lower = text.toLowerCase();
  const counts = {};
  let total = 0;

  BAD_WORDS.forEach(word => {
    const regex = new RegExp(`\\b${word}\\b`, 'g');
    const matches = lower.match(regex);
    if (matches) {
      counts[word] = matches.length;
      total += matches.length;
    }
  });

  return { total, counts };
}

function sendReport(pageUrl, pageTitle, badWordCount, badWordCounts) {
  if (!pageUrl) return;
  if (window.__badWordReport && window.__badWordReport.lastUrl === pageUrl && window.__badWordReport.count === badWordCount) {
    return;
  }

  window.__badWordReport = { lastUrl: pageUrl, count: badWordCount };

  fetch('http://127.0.0.1:5000/api/report', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      pageUrl,
      pageTitle,
      badWordCount,
      badWords: badWordCounts,
    })
  })
  .then(res => res.json())
  .then(data => {
    console.log('Bad word report saved:', data);
  })
  .catch(err => console.error('Error reporting bad words:', err));
}

function scanPageForBadWords() {
  const pageUrl = window.location.href;
  const pageTitle = document.title || pageUrl;
  const text = document.body.innerText || '';
  const { total, counts } = countBadWords(text);

  if (total > 0) {
    sendReport(pageUrl, pageTitle, total, counts);
  }
}

function scanCommentsForToxicity() {
  const comments = document.querySelectorAll('#content-text, .comment-renderer-text-content, .Comment, [data-test-id="comment"]');
  comments.forEach(comment => {
    if (comment.classList.contains('checked')) return;
    const text = comment.innerText.trim();
    if (!text) return;
    comment.classList.add('checked');

    fetch('http://127.0.0.1:5000/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    })
      .then(res => res.json())
      .then(scores => {
        const toxicScore = scores.toxic || 0;
        if (toxicScore > 0.6) {
          comment.style.backgroundColor = 'rgba(255, 0, 0, 0.2)';
          comment.style.borderLeft = '3px solid red';
          comment.title = `Toxicity: ${toxicScore.toFixed(2)} (Highly toxic)`;
        } else if (toxicScore > 0.3) {
          comment.style.backgroundColor = 'rgba(255, 165, 0, 0.18)';
          comment.style.borderLeft = '3px solid orange';
          comment.title = `Toxicity: ${toxicScore.toFixed(2)} (Moderate)`;
        } else {
          comment.style.backgroundColor = 'rgba(0, 255, 0, 0.12)';
          comment.style.borderLeft = '3px solid green';
          comment.title = `Toxicity: ${toxicScore.toFixed(2)} (Clean)`;
        }
      })
      .catch(err => console.error('Error connecting to Flask:', err));
  });
}

scanPageForBadWords();
setInterval(scanPageForBadWords, 10000);
setInterval(scanCommentsForToxicity, 5000);
