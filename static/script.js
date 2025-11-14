class TerrainAnalyzer {
    constructor() {
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        const uploadArea = document.getElementById('uploadArea');
        const imageInput = document.getElementById('imageInput');
        const selectBtn = document.getElementById('selectImageBtn');

        // Click sólo si se hace sobre el área vacía, no sobre hijos
        uploadArea.addEventListener('click', (e) => {
            if (e.target === uploadArea) {
                imageInput.click();
            }
        });

        // Botón dedicado
        if (selectBtn) {
            selectBtn.addEventListener('click', (e) => {
                e.stopPropagation(); // Evita que suba al uploadArea
                imageInput.click();
            });
        }

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleImageSelection(files[0]);
            }
        });

        // Cambio en input de archivo
        imageInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleImageSelection(e.target.files[0]);
            }
        });

        // Control de opacidad para la superposición
        const overlaySlider = document.getElementById('overlayOpacity');
        if (overlaySlider) {
            overlaySlider.addEventListener('input', (e) => {
                const overlayImg = document.getElementById('overlayPredImage');
                if (overlayImg) overlayImg.style.opacity = e.target.value;
            });
        }
    }

    handleImageSelection(file) {
        // Validar que sea imagen
        if (!file.type.startsWith('image/')) {
            this.showError('Por favor, selecciona un archivo de imagen válido.');
            return;
        }

        // Validar tamaño (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            this.showError('La imagen es demasiado grande. Máximo 10MB.');
            return;
        }

        // Mostrar preview y procesar
        this.showPreview(file);
        this.analyzeImage(file);
    }

    showPreview(file) {
        const reader = new FileReader();
        
        reader.onload = (e) => {
            document.getElementById('originalImage').src = e.target.result;
            this.showLoading();
        };
        
        reader.readAsDataURL(file);
    }

    async analyzeImage(file) {
        try {
            const formData = new FormData();
            formData.append('image', file);

            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                this.displayResults(result);
            } else {
                this.showError(result.error || 'Error al procesar la imagen.');
            }

        } catch (error) {
            this.showError('Error de conexión: ' + error.message);
        }
    }

    displayResults(result) {
        this.hideLoading();
        
        // Mostrar imágenes base
        document.getElementById('originalImage').src = result.original_image;
        document.getElementById('predictionImage').src = result.prediction_mask;

        // NUEVO: Asignar imágenes a la superposición
        const overlayOrig = document.getElementById('overlayOriginalImage');
        const overlayPred = document.getElementById('overlayPredImage');
        if (overlayOrig) overlayOrig.src = result.original_image;
        if (overlayPred) overlayPred.src = result.prediction_mask;

        // Actualizar estadísticas
        this.updateStatistics(result.statistics, result.percentages);

        // Mostrar sección de resultados
        document.getElementById('resultsSection').style.display = 'block';
        document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
    }

    updateStatistics(stats, percentages) {
        document.getElementById('normalStat').textContent = percentages.normal;
        document.getElementById('normalPixels').textContent = `${stats.normal.toLocaleString()} píxeles`;
        
        document.getElementById('inundacionStat').textContent = percentages.inundacion;
        document.getElementById('inundacionPixels').textContent = `${stats.inundacion.toLocaleString()} píxeles`;
        
        document.getElementById('depresionStat').textContent = percentages.depresion;
        document.getElementById('depresionPixels').textContent = `${stats.depresion.toLocaleString()} píxeles`;
    }

    showLoading() {
        document.getElementById('loading').style.display = 'block';
        document.getElementById('resultsSection').style.display = 'none';
        document.getElementById('error').style.display = 'none';
    }

    hideLoading() {
        document.getElementById('loading').style.display = 'none';
    }

    showError(message) {
        this.hideLoading();
        const errorElement = document.getElementById('error');
        document.getElementById('errorMessage').textContent = message;
        errorElement.style.display = 'block';
        
        // Ocultar error después de 5 segundos
        setTimeout(() => {
            errorElement.style.display = 'none';
        }, 5000);
    }
}

// Inicializar cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', () => {
    const analyzer = new TerrainAnalyzer();
    // Ejecutar prueba de mapeos aquí (antes estaba en otro listener duplicado)
    if (typeof testColorMappings === 'function') {
        testColorMappings();
    }
});

// Efectos adicionales
document.addEventListener('DOMContentLoaded', () => {
    // Animación de entrada para las tarjetas de estadísticas
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observar elementos para animar
    document.querySelectorAll('.stat-card').forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(card);
    });
});

// Función para probar mapeos (agregar al final del archivo)
function testColorMappings() {
    fetch('/test_mapping')
        .then(response => response.json())
        .then(data => {
            console.log('🔍 Resultados de test de mapeo:');
            data.test_results.forEach(result => {
                console.log(`   ${result.name}`);
                console.log(`   Distribución: Normal ${result.distribution.normal}, Inundación ${result.distribution.inundacion}, Depresión ${result.distribution.depresion}`);
            });
        })
        .catch(error => console.error('Error testing mappings:', error));
}