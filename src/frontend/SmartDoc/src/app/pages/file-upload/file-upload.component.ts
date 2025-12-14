import { ChangeDetectorRef, Component, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';

@Component({
  selector: 'app-file-upload',
  standalone: true,
  imports: [CommonModule, RouterLink],
  templateUrl: './file-upload.component.html',
  styleUrls: ['./file-upload.component.scss'],
})
export class FileUploadComponent {
  selectedFile: File | null = null;
  previewUrl: string | ArrayBuffer | null = null;
  isDragOver = false;
  private cdr = inject(ChangeDetectorRef);

  onFileSelected(event: any): void {
    const file = event.target.files[0];
    if (file) {
      this.handleFile(file);
    }
  }

  onDragOver(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDragOver = true;
  }

  onDragLeave(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDragOver = false;
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDragOver = false;

    if (event.dataTransfer?.files.length) {
      this.handleFile(event.dataTransfer.files[0]);
    }
  }

  handleFile(file: File): void {
    const validZipTypes = ['application/zip', 'application/x-zip-compressed', 'application/x-zip'];

    const isValidType = validZipTypes.includes(file.type) || file.name.endsWith('.zip');

    if (!isValidType) {
      alert('ZIP files only are supported.');
      return;
    }

    this.selectedFile = file;

    // Vorschau erstellen
    const reader = new FileReader();
    reader.onload = (e) => {
      this.previewUrl = e.target?.result ?? null;
      this.cdr.detectChanges();
    };
    reader.readAsDataURL(file);
  }

  upload(): void {
    if (!this.selectedFile) return;

    console.log('Uploading file to backend...', this.selectedFile.name);
  }

  reset(): void {
    this.selectedFile = null;
    this.previewUrl = null;
  }
}
