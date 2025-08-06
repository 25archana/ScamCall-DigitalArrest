const ctx = document.getElementById('scamChart').getContext('2d');
new Chart(ctx, {
    type: 'bar',
    data: {
        labels: ['Bank Scams', 'Call Scams', 'Online Fraud'],
        datasets: [{
            label: 'Scam Cases',
            data: [1500, 2200, 1300],
            backgroundColor: ['red', 'blue', 'green']
        }]
    }
});

document.addEventListener("DOMContentLoaded", function() {
    gsap.from(".hero h1", { opacity: 0, y: -50, duration: 1 });
    gsap.from(".hero p", { opacity: 0, y: -20, duration: 1, delay: 0.5 });
    gsap.from(".cta-btn", { opacity: 0, scale: 0.8, duration: 0.8, delay: 1 });
    
    gsap.from(".awareness-section", { opacity: 0, y: 50, duration: 1, scrollTrigger: ".awareness-section" });
    gsap.from(".news-survey", { opacity: 0, y: 50, duration: 1, scrollTrigger: ".news-survey" });
    gsap.from(".dataset-section", { opacity: 0, y: 50, duration: 1, scrollTrigger: ".dataset-section" });
});




document.addEventListener("DOMContentLoaded", function () {
    var videoModal = document.getElementById('videoModal1');
    var videoIframe = document.getElementById('youtubeVideo');

    videoModal.addEventListener('hidden.bs.modal', function () {
        // Reset the iframe src to stop the video
        videoIframe.src = videoIframe.src;
    });
});